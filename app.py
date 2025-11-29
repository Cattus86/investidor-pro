import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Enterprise", layout="wide")

st.title("🚀 Investidor Pro: Enterprise Edition")
st.markdown("### Plataforma de Análise Fundamentalista")

# --- MOTOR DE DADOS & LIMPEZA ---

@st.cache_data(ttl=300)
def carregar_dados():
    try:
        # 1. Busca dados brutos (Raw)
        df = fundamentus.get_resultado_raw()
        df = df.reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # 2. Padronização de Colunas (Minúsculo e sem caracteres especiais)
        df.columns = [c.replace('.', '').replace('/', '').replace(' ', '').lower() for c in df.columns]
        
        # 3. Mapa de Tradução (Do Fundamentus para nosso Padrão)
        mapa = {
            'papel': 'Ticker',
            'ticker': 'Ticker',
            'cotacao': 'Preco',
            'pl': 'PL',
            'pvp': 'PVP',
            'dy': 'DY',
            'divyield': 'DY',
            'roe': 'ROE',
            'roic': 'ROIC',
            'evebit': 'EV_EBIT',
            'liq2meses': 'Liquidez',
            'liq2m': 'Liquidez',
            'mrglíq': 'MargemLiquida',
            'mrgliq': 'MargemLiquida',
            'divbrutpatr': 'Div_Patrimonio'
        }
        
        # Renomeia o que encontrar
        df = df.rename(columns=mapa)
        
        # 4. Forçar Numérico (O Segredo para não travar)
        cols_numericas = ['Preco', 'PL', 'PVP', 'DY', 'ROE', 'ROIC', 'EV_EBIT', 'Liquidez', 'MargemLiquida', 'Div_Patrimonio']
        
        for col in cols_numericas:
            if col not in df.columns:
                df[col] = 0.0 # Cria coluna zerada se não existir
            
            # Força conversão para números (erros viram NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        # 5. Ajuste de Escala (Decimal para Percentual)
        # Se a média do DY for menor que 1 (ex: 0.06), multiplica por 100
        if df['DY'].mean() < 1: df['DY'] = df['DY'] * 100
        if df['ROE'].mean() < 1: df['ROE'] = df['ROE'] * 100
        if df['ROIC'].mean() < 1: df['ROIC'] = df['ROIC'] * 100
        
        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# --- CÁLCULOS ESTRATÉGICOS ---
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

    # Bazin (Preço Teto 6%)
    df['Bazin_Teto'] = np.where(df['DY'] > 0, df['Preco'] * (df['DY'] / 6), 0)
    
    return df

# --- INTERFACE ---
with st.spinner('Analisando mercado...'):
    df_raw = carregar_dados()

if not df_raw.empty:
    df = processar_estrategias(df_raw)
    
    # Barra Lateral
    st.sidebar.header("Filtros")
    liquidez_min = st.sidebar.number_input("Liquidez Mínima (R$)", value=200000.0, step=100000.0)
    df = df[df['Liquidez'] >= liquidez_min]

    # Abas
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Dividendos", "💎 Graham", "✨ Magic Formula", "📈 Gráfico"])

    # --- ABA DIVIDENDOS ---
    with tab1:
        st.subheader("Top Dividendos (Bazin)")
        cols_bazin = ['Ticker', 'Preco', 'DY', 'PVP', 'Bazin_Teto']
        df_bazin = df.nlargest(20, 'DY')
        
        # Tenta aplicar estilo, se falhar, mostra tabela normal (FAIL-SAFE)
        try:
            st.dataframe(
                df_bazin[cols_bazin].set_index('Ticker').style
                .format({'Preco': 'R$ {:.2f}', 'DY': '{:.2f}%', 'PVP': '{:.2f}', 'Bazin_Teto': 'R$ {:.2f}'})
                .background_gradient(subset=['DY'], cmap='Greens'),
                use_container_width=True
            )
        except:
            st.warning("Modo de exibição simplificado (Erro de renderização gráfica).")
            st.dataframe(df_bazin[cols_bazin].set_index('Ticker'), use_container_width=True)

    # --- ABA GRAHAM ---
    with tab2:
        st.subheader("Ranking Graham (Preço Justo)")
        cols_graham = ['Ticker', 'Preco', 'Graham_Preco', 'Graham_Upside', 'PL', 'PVP']
        df_graham = df[(df['Graham_Upside'] > 0) & (df['Graham_Upside'] < 500)].nlargest(20, 'Graham_Upside')
        
        try:
            st.dataframe(
                df_graham[cols_graham].set_index('Ticker').style
                .format({'Preco': 'R$ {:.2f}', 'Graham_Preco': 'R$ {:.2f}', 'Graham_Upside': '{:.2f}%', 'PL': '{:.2f}'})
                .bar(subset=['Graham_Upside'], color='lightgreen'),
                use_container_width=True
            )
        except:
            st.dataframe(df_graham[cols_graham].set_index('Ticker'), use_container_width=True)

    # --- ABA MAGIC FORMULA ---
    with tab3:
        st.subheader("Magic Formula (Greenblatt)")
        cols_magic = ['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']
        df_magic_view = df.nsmallest(20, 'Score_Magic')
        
        try:
            st.dataframe(
                df_magic_view[cols_magic].set_index('Ticker').style
                .format({'Preco': 'R$ {:.2f}', 'EV_EBIT': '{:.2f}', 'ROIC': '{:.2f}%', 'Score_Magic': '{:.0f}'})
                .background_gradient(subset=['ROIC'], cmap='Blues'),
                use_container_width=True
            )
        except:
            st.dataframe(df_magic_view[cols_magic].set_index('Ticker'), use_container_width=True)

    # --- ABA GRÁFICO ---
    with tab4:
        st.subheader("Mapa de Oportunidades")
        df_chart = df[(df['PL'] > 0) & (df['PL'] < 40) & (df['ROE'] > 0) & (df['ROE'] < 50)]
        fig = px.scatter(df_chart, x='PL', y='ROE', color='DY', size='Liquidez', 
                         hover_name='Ticker', title="P/L vs ROE (Cor = Dividendos)")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Erro crítico: Não foi possível baixar os dados. Verifique o requirements.txt.")
