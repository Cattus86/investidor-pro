import streamlit as st
import pandas as pd
import fundamentus
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Quant", layout="wide")
st.title("⚡ Investidor Pro: Plataforma Quantitativa")
st.markdown("Análise fundamentalista automatizada de todos os ativos da B3.")

# --- BARRA LATERAL ---
st.sidebar.header("🔍 Filtros Globais")
min_liquidez = st.sidebar.number_input("Liquidez Diária Mínima (R$):", value=200000, step=50000)

# --- MOTOR DE DADOS (Versão 5.0 - Compatível com seu Print) ---
@st.cache_data(ttl=3600)
def carregar_base_completa():
    try:
        # 1. Baixar dados
        df = fundamentus.get_resultado()
        
        # 2. Resetar Index (O Ticker sai do index e vira coluna 'papel')
        df = df.reset_index()
        
        # 3. NORMALIZAÇÃO RADICAL
        # Transforma TUDO em minúsculo e remove pontos/barras para padronizar
        # Ex: "Liq.2m" vira "liq2m", "P/L" vira "pl", "Papel" vira "papel"
        df.columns = [col.replace('.', '').replace('/', '').replace(' ', '').lower() for col in df.columns]
        
        # 4. TRADUÇÃO (Mapeamento baseado no seu print)
        rename_map = {
            'papel': 'Ticker',    # Nome original do index
            'ticker': 'Ticker',   # Caso já tenha vindo como ticker
            'cotacao': 'Preco',
            'pl': 'PL',
            'pvp': 'PVP',
            'dy': 'DY',           # Às vezes vem dy
            'divyield': 'DY',     # Às vezes vem divyield
            'evebit': 'EV_EBIT',
            'evebitda': 'EV_EBIT', # Fallback
            'mrgebit': 'MargemEbit',
            'mrgliq': 'MargemLiquida',
            'roic': 'ROIC',
            'roe': 'ROE',
            'liq2m': 'Liquidez',      # IDENTIFICADO NO SEU PRINT
            'liq2meses': 'Liquidez'   # Antigo padrão
        }
        
        # Aplica a renomeação
        df = df.rename(columns=rename_map)
        
        # 5. GARANTIA DE COLUNAS (Cria se não existir para não travar)
        cols_essenciais = ['Preco', 'PL', 'PVP', 'DY', 'ROE', 'ROIC', 'EV_EBIT', 'Liquidez']
        for col in cols_essenciais:
            if col not in df.columns:
                df[col] = 0

        # 6. CONVERSÃO DE TIPOS (Texto -> Número)
        cols_percentuais = ['DY', 'ROE', 'ROIC', 'MargemLiquida']
        for col in cols_percentuais:
            df[col] = pd.to_numeric(df[col], errors='coerce') * 100

        cols_numericas = ['Preco', 'PL', 'PVP', 'EV_EBIT', 'Liquidez']
        for col in cols_numericas:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- MOTOR DE CÁLCULO ---
        
        # Graham
        # LPA = Preço / PL | VPA = Preço / PVP
        # Evita divisão por zero
        df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
        df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
        
        def calcular_graham(row):
            if row['LPA'] > 0 and row['VPA'] > 0:
                return np.sqrt(22.5 * row['LPA'] * row['VPA'])
            return 0
            
        df['Preco_Justo_Graham'] = df.apply(calcular_graham, axis=1)
        df['Potencial_Graham'] = np.where(
            (df['Preco_Justo_Graham'] > 0) & (df['Preco'] > 0),
            ((df['Preco_Justo_Graham'] - df['Preco']) / df['Preco']) * 100,
            -999
        )

        # Magic Formula
        # Agora usamos 'EV_EBIT' e 'ROIC' garantidos
        df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
        if not df_magic.empty:
            df_magic['Rank_EV_EBIT'] = df_magic['EV_EBIT'].rank(ascending=True)
            df_magic['Rank_ROIC'] = df_magic['ROIC'].rank(ascending=False)
            df_magic['Score_Magic'] = df_magic['Rank_EV_EBIT'] + df_magic['Rank_ROIC']
            df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999

        return df

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
        # Debug visual para ajudar se der erro de novo
        try:
            raw = fundamentus.get_resultado().reset_index()
            raw.columns = [c.lower().replace('.','') for c in raw.columns]
            st.write("Estrutura recebida (Debug):", raw.columns.tolist())
        except:
            pass
        return pd.DataFrame()

# --- INTERFACE ---
with st.spinner('Processando dados...'):
    df = carregar_base_completa()

if not df.empty and 'Liquidez' in df.columns:
    # Filtro de Liquidez
    df = df[df['Liquidez'] >= min_liquidez].copy()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Visão Geral", "💰 Dividendos", "⚖️ Graham", "✨ Magic Formula"])

    with tab1:
        st.subheader("Mercado Completo")
        cols = ['Ticker', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'Liquidez']
        st.dataframe(df[cols].set_index('Ticker'), use_container_width=True)

    with tab2:
        st.subheader("Top Dividendos")
        top_div = df.sort_values(by='DY', ascending=False).head(20)
        st.dataframe(top_div[['Ticker', 'Preco', 'DY', 'PVP']].style.format({'Preco':'R$ {:.2f}', 'DY':'{:.2f}%', 'PVP':'{:.2f}'}), use_container_width=True)

    with tab3:
        st.subheader("Ranking Graham")
        graham = df[df['Potencial_Graham'] > 0].sort_values(by='Potencial_Graham', ascending=False).head(30)
        st.dataframe(graham[['Ticker', 'Preco', 'Preco_Justo_Graham', 'Potencial_Graham']].style.format({'Preco':'R$ {:.2f}', 'Preco_Justo_Graham':'R$ {:.2f}', 'Potencial_Graham':'{:.2f}%'}), use_container_width=True)

    with tab4:
        st.subheader("Magic Formula")
        if 'Score_Magic' in df.columns:
            magic = df.dropna(subset=['Score_Magic']).sort_values(by='Score_Magic').head(30)
            st.dataframe(magic[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']].style.format({'Preco':'R$ {:.2f}', 'EV_EBIT':'{:.2f}', 'ROIC':'{:.2f}%', 'Score_Magic':'{:.0f}'}), use_container_width=True)

else:
    st.warning("Carregando base de dados...")
