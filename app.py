import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Enterprise", layout="wide")
st.title("🚀 Investidor Pro: Enterprise Edition")

# --- MOTOR DE DADOS ---
@st.cache_data(ttl=300)
def carregar_dados():
    try:
        # 1. Coleta e Tratamento Inicial
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # 2. Padronização de Colunas
        df.columns = [c.replace('.', '').replace('/', '').replace(' ', '').lower() for c in df.columns]
        
        mapa = {
            'papel': 'Ticker', 'ticker': 'Ticker', 'cotacao': 'Preco',
            'pl': 'PL', 'pvp': 'PVP', 'dy': 'DY', 'divyield': 'DY',
            'roe': 'ROE', 'roic': 'ROIC', 'evebit': 'EV_EBIT',
            'liq2meses': 'Liquidez', 'liq2m': 'Liquidez',
            'mrglíq': 'MargemLiquida', 'mrgliq': 'MargemLiquida'
        }
        df = df.rename(columns=mapa)
        
        # 3. Forçar Numérico e Limpar
        cols_num = ['Preco', 'PL', 'PVP', 'DY', 'ROE', 'ROIC', 'EV_EBIT', 'Liquidez', 'MargemLiquida']
        for col in cols_num:
            if col not in df.columns: df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        # 4. Ajustes Percentuais
        if df['DY'].mean() < 1: df['DY'] *= 100
        if df['ROE'].mean() < 1: df['ROE'] *= 100
        if df['ROIC'].mean() < 1: df['ROIC'] *= 100
        
        return df
    except Exception as e:
        st.error(f"Erro no motor de dados: {e}")
        return pd.DataFrame()

# --- MOTOR DE ESTRATÉGIAS ---
def processar_estrategias(df):
    # Graham
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    
    mask_graham = (df['LPA'] > 0) & (df['VPA'] > 0)
    df.loc[mask_graham, 'Graham_Preco'] = np.sqrt(22.5 * df.loc[mask_graham, 'LPA'] * df.loc[mask_graham, 'VPA'])
    df['Graham_Preco'] = df['Graham_Preco'].fillna(0)
    
    df['Graham_Upside'] = np.where(
        (df['Graham_Preco'] > 0) & (df['Preco'] > 0),
        ((df['Graham_Preco'] - df['Preco']) / df['Preco']) * 100,
        -999
    )

    # Magic Formula
    df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
    if not df_magic.empty:
        df_magic['Rank_EV'] = df_magic['EV_EBIT'].rank(ascending=True)
        df_magic['Rank_ROIC'] = df_magic['ROIC'].rank(ascending=False)
        df_magic['Score_Magic'] = df_magic['Rank_EV'] + df_magic['Rank_ROIC']
        df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
    else:
        df['Score_Magic'] = 99999

    # Bazin (Teto 6%)
    df['Bazin_Teto'] = np.where(df['DY'] > 0, df['Preco'] * (df['DY'] / 6), 0)
    
    return df

# --- FRONT-END (INTERFACE MODERNA) ---
with st.spinner('Processando indicadores...'):
    df_raw = carregar_dados()

if not df_raw.empty:
    df = processar_estrategias(df_raw)
    
    # Filtro Lateral
    st.sidebar.header("Filtros")
    liq_min = st.sidebar.number_input("Liquidez Mínima (R$)", value=200000.0, step=100000.0)
    df_view = df[df['Liquidez'] >= liq_min].copy()

    # Abas
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Dividendos", "💎 Graham", "✨ Magic Formula", "📈 Gráficos"])

    # 1. DIVIDENDOS (Com Barra de Progresso Visual)
    with tab1:
        st.subheader("Top Dividendos (Bazin)")
        df_bazin = df_view.nlargest(20, 'DY')
        
        st.dataframe(
            df_bazin[['Ticker', 'Preco', 'DY', 'PVP', 'Bazin_Teto']],
            column_config={
                "Preco": st.column_config.NumberColumn("Preço Atual", format="R$ %.2f"),
                "DY": st.column_config.ProgressColumn(
                    "Dividend Yield", 
                    format="%.2f%%", 
                    min_value=0, 
                    max_value=20, # Barra cheia em 20%
                    help="Quanto maior a barra, maior o dividendo."
                ),
                "Bazin_Teto": st.column_config.NumberColumn("Preço Teto (6%)", format="R$ %.2f"),
                "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

    # 2. GRAHAM (Com Barra de Potencial)
    with tab2:
        st.subheader("Ranking Graham (Desconto)")
        df_graham = df_view[(df_view['Graham_Upside'] > 0) & (df_view['Graham_Upside'] < 300)].nlargest(20, 'Graham_Upside')
        
        st.dataframe(
            df_graham[['Ticker', 'Preco', 'Graham_Preco', 'Graham_Upside', 'PL', 'PVP']],
            column_config={
                "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
                "Graham_Preco": st.column_config.NumberColumn("Preço Justo", format="R$ %.2f"),
                "Graham_Upside": st.column_config.ProgressColumn(
                    "Potencial (%)", 
                    format="%.1f%%", 
                    min_value=0, 
                    max_value=100
                ),
                "PL": st.column_config.NumberColumn("P/L", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )

    # 3. MAGIC FORMULA
    with tab3:
        st.subheader("Magic Formula (Qualidade + Preço)")
        df_magic = df_view.nsmallest(20, 'Score_Magic')
        
        st.dataframe(
            df_magic[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']],
            column_config={
                "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
                "ROIC": st.column_config.ProgressColumn("ROIC (Qualidade)", format="%.1f%%", min_value=0, max_value=50),
                "EV_EBIT": st.column_config.NumberColumn("EV/EBIT (Preço)", format="%.2f"),
                "Score_Magic": st.column_config.NumberColumn("Score (Menor = Melhor)"),
            },
            hide_index=True,
            use_container_width=True
        )

    # 4. GRÁFICOS
    with tab4:
        st.subheader("Raio-X do Mercado")
        df_chart = df_view[(df_view['PL'] > 0) & (df_view['PL'] < 40) & (df_view['ROE'] > 0) & (df_view['ROE'] < 50)]
        
        fig = px.scatter(
            df_chart, x='PL', y='ROE', color='DY', size='Liquidez',
            hover_name='Ticker', 
            title="Mapa: P/L vs ROE (Cor = Dividendos)",
            labels={'PL': 'P/L (Anos)', 'ROE': 'ROE (%)', 'DY': 'Yield (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Aguardando dados... Se demorar, recarregue a página.")
