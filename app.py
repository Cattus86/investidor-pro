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
min_liquidez = st.sidebar.number_input("Liquidez Diária Mínima (R$):", value=200000.0, step=50000.0)

# --- MOTOR DE DADOS (Versão 6.0 - Mapeamento Estrito) ---
@st.cache_data(ttl=3600)
def carregar_base_completa():
    try:
        # 1. Baixar dados brutos
        df_raw = fundamentus.get_resultado()
        
        # 2. Resetar Index para trazer o Ticker para dentro
        df_raw = df_raw.reset_index()
        
        # 3. Limpeza Radical dos Nomes das Colunas
        # Transforma tudo em minúsculo para evitar erros de 'Papel' vs 'papel'
        df_raw.columns = df_raw.columns.str.lower()
        
        # 4. CRIAÇÃO DE UM NOVO DATAFRAME LIMPO
        # Em vez de renomear o antigo, vamos criar um novo só com o que importa.
        # Isso evita o erro de "colunas duplicadas" ou lixo de memória.
        
        df = pd.DataFrame()
        
        # Mapeamento Manual baseado no seu Print
        # Coluna no Fundamentus -> Coluna no nosso App
        mapa = {
            'papel': 'Ticker',
            'cotacao': 'Preco',
            'pl': 'PL',
            'pvp': 'PVP',
            'dy': 'DY',
            'evebit': 'EV_EBIT',
            'roic': 'ROIC',
            'roe': 'ROE',
            'liq2m': 'Liquidez'  # O culpado estava aqui (às vezes vem liq2meses)
        }
        
        # Copia apenas as colunas que existem
        for col_origem, col_destino in mapa.items():
            if col_origem in df_raw.columns:
                df[col_destino] = df_raw[col_origem]
            else:
                # Se não existir (ex: dy veio como divyield), tenta alternativas
                if col_origem == 'dy' and 'divyield' in df_raw.columns:
                    df['DY'] = df_raw['divyield']
                elif col_origem == 'liq2m' and 'liq2meses' in df_raw.columns:
                    df['Liquidez'] = df_raw['liq2meses']
                else:
                    df[col_destino] = 0 # Preenche com 0 se não achar
        
        # 5. CONVERSÃO NÚMERICA SEGURA
        # Removemos qualquer caractere estranho e forçamos virar número
        cols_numericas = ['Preco', 'PL', 'PVP', 'EV_EBIT', 'ROIC', 'ROE', 'DY', 'Liquidez']
        
        for col in cols_numericas:
            # errors='coerce' transforma textos ruins em NaN (Not a Number)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # Ajuste de Porcentagem (0.15 -> 15.0) para DY, ROE, ROIC
        # Mas verificamos se os dados já não vieram multiplicados (algumas versões vêm)
        # Assumindo padrão < 1 é decimal
        if df['DY'].mean() < 1: df['DY'] = df['DY'] * 100
        if df['ROE'].mean() < 1: df['ROE'] = df['ROE'] * 100
        if df['ROIC'].mean() < 1: df['ROIC'] = df['ROIC'] * 100

        # 6. CÁLCULOS FINANCEIROS (Graham & Greenblatt)
        
        # Graham
        # Evitar divisão por zero
        df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
        df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
        
        # Graham só funciona se LPA e VPA forem positivos
        mask_graham = (df['LPA'] > 0) & (df['VPA'] > 0)
        df.loc[mask_graham, 'Preco_Justo_Graham'] = np.sqrt(22.5 * df.loc[mask_graham, 'LPA'] * df.loc[mask_graham, 'VPA'])
        df['Preco_Justo_Graham'] = df['Preco_Justo_Graham'].fillna(0)
        
        mask_potencial = (df['Preco_Justo_Graham'] > 0) & (df['Preco'] > 0)
        df.loc[mask_potencial, 'Potencial_Graham'] = ((df['Preco_Justo_Graham'] - df['Preco']) / df['Preco']) * 100
        df['Potencial_Graham'] = df['Potencial_Graham'].fillna(-999)

        # Magic Formula
        # Filtra empresas com dados operacionais válidos
        df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
        
        if not df_magic.empty:
            df_magic['Rank_EV_EBIT'] = df_magic['EV_EBIT'].rank(ascending=True)
            df_magic['Rank_ROIC'] = df_magic['ROIC'].rank(ascending=False)
            df_magic['Score_Magic'] = df_magic['Rank_EV_EBIT'] + df_magic['Rank_ROIC']
            
            # Traz o score de volta para o dataframe principal
            df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = None

        return df

    except Exception as e:
        st.error(f"Erro Crítico: {e}")
        # Debug para ajudar a rastrear
        try:
            st.write("Colunas recebidas:", fundamentus.get_resultado().reset_index().columns.tolist())
        except:
            pass
        return pd.DataFrame()

# --- EXECUÇÃO ---
with st.spinner('Processando dados do mercado...'):
    df = carregar_base_completa()

if not df.empty and 'Liquidez' in df.columns:
    
    # Filtro de Liquidez
    df_filtrado = df[df['Liquidez'] >= min_liquidez].copy()
    
    # --- VISUALIZAÇÃO ---
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Visão Geral", "💰 Dividendos", "💎 Graham", "✨ Magic Formula"])
    
    with tab1:
        st.subheader("Visão Geral do Mercado")
        cols_view = ['Ticker', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'Liquidez']
        st.dataframe(
            df_filtrado[cols_view].set_index('Ticker').style.format({
                'Preco': 'R$ {:.2f}', 'PL': '{:.2f}', 'PVP': '{:.2f}', 
                'DY': '{:.2f}%', 'ROE': '{:.2f}%', 'Liquidez': 'R$ {:,.0f}'
            }), 
            use_container_width=True
        )

    with tab2:
        st.subheader("Top Pagadoras de Dividendos")
        top_dy = df_filtrado.nlargest(20, 'DY')
        st.dataframe(
            top_dy[['Ticker', 'Preco', 'DY', 'PVP']].set_index('Ticker').style
            .format({'Preco': 'R$ {:.2f}', 'DY': '{:.2f}%', 'PVP': '{:.2f}'})
            .background_gradient(subset=['DY'], cmap='Greens'),
            use_container_width=True
        )

    with tab3:
        st.subheader("Oportunidades Graham (Preço Justo)")
        # Filtra apenas quem tem potencial positivo
        df_graham = df_filtrado[df_filtrado['Potencial_Graham'] > 0].sort_values('Potencial_Graham', ascending=False).head(30)
        
        st.dataframe(
            df_graham[['Ticker', 'Preco', 'Preco_Justo_Graham', 'Potencial_Graham']].set_index('Ticker').style
            .format({'Preco': 'R$ {:.2f}', 'Preco_Justo_Graham': 'R$ {:.2f}', 'Potencial_Graham': '{:.2f}%'})
            .bar(subset=['Potencial_Graham'], color='lightgreen'),
            use_container_width=True
        )

    with tab4:
        st.subheader("Ranking Magic Formula")
        # Filtra quem tem Score Magic
        df_magic_view = df_filtrado.dropna(subset=['Score_Magic']).sort_values('Score_Magic', ascending=True).head(30)
        
        st.dataframe(
            df_magic_view[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']].set_index('Ticker').style
            .format({'Preco': 'R$ {:.2f}', 'EV_EBIT': '{:.2f}', 'ROIC': '{:.2f}%', 'Score_Magic': '{:.0f}'})
            .background_gradient(subset=['Score_Magic'], cmap='Blues_r'),
            use_container_width=True
        )

else:
    st.warning("Aguardando dados... Se o erro persistir, verifique a conexão.")
