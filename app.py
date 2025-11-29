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

# --- MOTOR DE DADOS BLINDADO ---
@st.cache_data(ttl=3600)
def carregar_base_completa():
    try:
        # 1. Baixar dados brutos
        df = fundamentus.get_resultado()
        df = df.reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # 2. LIMPEZA DE COLUNAS (A Solução Definitiva)
        # Transforma " P/L " em "pl", "Div.Yield" em "divyield"
        df.columns = [col.replace('.', '').replace('/', '').replace(' ', '').lower() for col in df.columns]
        
        # Agora sabemos exatamente como as colunas se chamam:
        # cotação -> cotacao (ou algo similar, vamos tratar abaixo)
        # p/l -> pl
        # p/vp -> pvp
        # div.yield -> divyield
        
        # Renomeia para nomes padronizados internos
        rename_map = {
            'cotacao': 'Preco',
            'cotação': 'Preco', # Garantia extra
            'pl': 'PL',
            'pvp': 'PVP',
            'divyield': 'DY',
            'roe': 'ROE',
            'roic': 'ROIC',
            'evebit': 'EV_EBIT',
            'liq2meses': 'Liquidez',
            'mrglíq': 'MargemLiquida',
            'mrliq': 'MargemLiquida'
        }
        
        # Renomeia o que encontrar
        df = df.rename(columns=rename_map)
        
        # 3. Tratamento de Tipos
        # Multiplica percentuais por 100
        cols_percentuais = ['DY', 'ROE', 'ROIC', 'MargemLiquida']
        for col in cols_percentuais:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') * 100

        # Garante números
        cols_numericas = ['Preco', 'PL', 'PVP', 'EV_EBIT', 'Liquidez']
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 4. CÁLCULOS AVANÇADOS
        
        # Graham (Precisa de PL e PVP)
        # Se PL ou PVP não existirem, cria com valor 0 para não travar
        if 'PL' not in df.columns: df['PL'] = 0
        if 'PVP' not in df.columns: df['PVP'] = 0
        if 'Preco' not in df.columns: df['Preco'] = 0

        # LPA = Preço / PL
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
        if 'EV_EBIT' in df.columns and 'ROIC' in df.columns:
            df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
            df_magic['Rank_EV_EBIT'] = df_magic['EV_EBIT'].rank(ascending=True)
            df_magic['Rank_ROIC'] = df_magic['ROIC'].rank(ascending=False)
            df_magic['Score_Magic'] = df_magic['Rank_EV_EBIT'] + df_magic['Rank_ROIC']
            df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999

        return df

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
        # MODO DEBUG: Se der erro, mostra as colunas que vieram para a gente saber o nome certo
        try:
            raw_data = fundamentus.get_resultado()
            st.write("Colunas encontradas na biblioteca (Debug):", raw_data.columns.tolist())
        except:
            pass
        return pd.DataFrame()

# --- CARREGAMENTO ---
with st.spinner('Processando Big Data da B3...'):
    df = carregar_base_completa()

if not df.empty and 'Liquidez' in df.columns:
    df = df[df['Liquidez'] >= min_liquidez].copy()
    
    # --- INTERFACE ---
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Visão Geral", "💰 Dividendos", "⚖️ Graham", "✨ Magic Formula"])

    with tab1:
        st.subheader("Mercado Completo")
        cols_view = [c for c in ['Ticker', 'Preco', 'PL', 'PVP', 'DY', 'ROE'] if c in df.columns]
        st.dataframe(df[cols_view].set_index('Ticker'), use_container_width=True)

    with tab2:
        st.subheader("Top Dividendos")
        if 'DY' in df.columns:
            df_div = df.sort_values(by='DY', ascending=False).head(20)
            st.dataframe(
                df_div[['Ticker', 'Preco', 'DY', 'PVP']].style
                .format({'Preco': 'R$ {:.2f}', 'DY': '{:.2f}%', 'PVP': '{:.2f}'})
                .background_gradient(subset=['DY'], cmap='Greens'),
                use_container_width=True
            )

    with tab3:
        st.subheader("Ranking Benjamin Graham")
        if 'Potencial_Graham' in df.columns:
            df_graham = df[df['Potencial_Graham'] > 0].sort_values(by='Potencial_Graham', ascending=False).head(30)
            st.dataframe(
                df_graham[['Ticker', 'Preco', 'Preco_Justo_Graham', 'Potencial_Graham']].style
                .format({'Preco': 'R$ {:.2f}', 'Preco_Justo_Graham': 'R$ {:.2f}', 'Potencial_Graham': '{:.2f}%'})
                .bar(subset=['Potencial_Graham'], color='lightgreen'),
                use_container_width=True
            )

    with tab4:
        st.subheader("Ranking Magic Formula")
        if 'Score_Magic' in df.columns:
            df_magic_view = df.dropna(subset=['Score_Magic']).sort_values(by='Score_Magic', ascending=True).head(30)
            st.dataframe(
                df_magic_view[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']].style
                .format({'Preco': 'R$ {:.2f}', 'EV_EBIT': '{:.2f}', 'ROIC': '{:.2f}%', 'Score_Magic': '{:.0f}'})
                .background_gradient(subset=['Score_Magic'], cmap='Blues_r'),
                use_container_width=True
            )
else:
    st.warning("Aguardando processamento...")
