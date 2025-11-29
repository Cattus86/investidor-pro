import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA (Layout Profissional) ---
st.set_page_config(page_title="Investidor Pro | Enterprise", layout="wide", initial_sidebar_state="expanded")

# CSS para remover bordas vazias e melhorar visual
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Investidor Pro: Enterprise Edition")
st.markdown("### A Plataforma Definitiva de Análise Fundamentalista")

# --- MOTOR DE DADOS & LIMPEZA (O Coração do Sistema) ---

def tratar_coluna_ptbr(serie):
    """
    Função robusta para converter padrão brasileiro (1.000,00) para float (1000.0).
    Remove %, pontos de milhar e converte vírgula em ponto.
    """
    if serie.dtype == 'float64' or serie.dtype == 'int64':
        return serie
    
    # Converte para string, remove espaços
    serie = serie.astype(str).str.strip()
    
    # Remove % e pontos de milhar (ex: 1.200 -> 1200)
    serie = serie.str.replace('.', '', regex=False)
    serie = serie.str.replace('%', '', regex=False)
    
    # Substitui vírgula decimal por ponto (ex: 12,5 -> 12.5)
    serie = serie.str.replace(',', '.', regex=False)
    
    # Converte para número, transformando erros em NaN (0)
    return pd.to_numeric(serie, errors='coerce').fillna(0)

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados_b3():
    try:
        # 1. Coleta Bruta
        df = fundamentus.get_resultado_raw() # Tenta pegar RAW para evitar formatações automáticas ruins
        
        # Reset index para ter a coluna Ticker
        df = df.reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # 2. Renomeação Inteligente (Padronização)
        # Transforma nomes estranhos em nomes padrão
        mapa_colunas = {
            'Cotação': 'Preco',
            'P/L': 'PL',
            'P/VP': 'PVP',
            'Div.Yield': 'DY',
            'ROE': 'ROE',
            'ROIC': 'ROIC',
            'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez',
            'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Divida_Patrimonio',
            'Cresc. Rec.5a': 'Crescimento5a'
        }
        
        # Renomeia colunas que existem no dataframe
        cols_existentes = {k: v for k, v in mapa_colunas.items() if k in df.columns}
        df = df.rename(columns=cols_existentes)
        
        # 3. Sanitização (A Correção do Erro)
        # Aplica a função de limpeza em TODAS as colunas, exceto Ticker
        for col in df.columns:
            if col != 'Ticker':
                df[col] = tratar_coluna_ptbr(df[col])

        # 4. Ajustes Percentuais
        # O fundamentus raw às vezes entrega 0.15 para 15%. Vamos ajustar.
        # Regra heurística: se a média da coluna for menor que 5, provavelmente está em decimal.
        cols_percent = ['DY', 'ROE', 'ROIC', 'MargemLiquida']
        for col in cols_percent:
            if col in df.columns:
                if df[col].mean() < 5: 
                    df[col] = df[col] * 100

        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados brutos: {e}")
        return pd.DataFrame()

# --- CÁLCULOS DE RANKING (Estratégias) ---

def calcular_rankings(df):
    # Graham (Preço Justo)
    # V = Raiz(22.5 * LPA * VPA). Mas como temos P/L e P/VP:
    # LPA = Preço / PL | VPA = Preço / PVP
    
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    
    # Graham só vale para empresas com lucro e patrimônio positivos
    mask_graham = (df['LPA'] > 0) & (df['VPA'] > 0)
    
    df.loc[mask_graham, 'Graham_Preco_Justo'] = np.sqrt(22.5 * df.loc[mask_graham, 'LPA'] * df.loc[mask_graham, 'VPA'])
    df['Graham_Preco_Justo'] = df['Graham_Preco_Justo'].fillna(0)
    
    df['Graham_Potencial'] = np.where(
        (df['Graham_Preco_Justo'] > 0) & (df['Preco'] > 0),
        ((df['Graham_Preco_Justo'] - df['Preco']) / df['Preco']) * 100,
        -999 # Joga pro final se não tiver potencial
    )

    # Magic Formula (Greenblatt Simplificada)
    # Ranking conjunto de EV/EBIT (Barato) + ROIC (Qualidade)
    if 'EV_EBIT' in df.columns and 'ROIC' in df.columns:
        # Remove negativos e zeros
        validos = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].index
        
        df.loc[validos, 'Rank_EV'] = df.loc[validos, 'EV_EBIT'].rank(ascending=True)
        df.loc[validos, 'Rank_ROIC'] = df.loc[validos, 'ROIC'].rank(ascending=False)
        df.loc[validos, 'Score_Magic'] = df.loc[validos, 'Rank_EV'] + df.loc[validos, 'Rank_ROIC']
    else:
        df['Score_Magic'] = 99999

    # Décio Bazin (Foco em Dividendos Seguros)
    # Preço Teto = DY Médio esperado (6%) -> Preço Teto = (Dividendos por Ação) / 0.06
    # Vamos simplificar usando o DY atual para projetar
    if 'DY' in df.columns:
        df['Bazin_Preco_Teto'] = np.where(df['DY'] > 0, df['Preco'] * (df['DY'] / 6), 0)
    
    return df

# --- INTERFACE PRINCIPAL ---

with st.spinner('Conectando ao Big Data da B3 e processando indicadores...'):
    df_raw = carregar_dados_b3()

if not df_raw.empty:
    df = calcular_rankings(df_raw.copy())

    # --- SIDEBAR: FILTROS PODEROSOS ---
    st.sidebar.header("🔍 Filtros de Mercado")
    
    # Filtro de Liquidez
    min_liq = st.sidebar.number_input("Liquidez Diária Mínima (R$)", value=200000.0, step=100000.0, format="%.0f")
    df = df[df['Liquidez'] >= min_liq]

    # Filtro de Setores/Tickers (Busca)
    busca = st.sidebar.text_input("Buscar Ativo (ex: PETR, VALE):").upper()
    if busca:
        df = df[df['Ticker'].str.contains(busca)]

    st.sidebar.markdown("---")
    st.sidebar.caption("Dados fornecidos por Fundamentus. Atraso de 15min.")

    # --- ABAS DE ESTRATÉGIA ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Dividendos (Bazin)", 
        "💎 Valor (Graham)", 
        "✨ Fórmula Mágica", 
        "🚀 Crescimento & Qualidade",
        "📊 Raio-X Geral"
    ])

    # 1. ABA DIVIDENDOS (Bazin)
    with tab1:
        st.subheader("O Método Bazin: Dividendos Sustentáveis")
        st.markdown("Filtra empresas que pagam acima de **6% de Dividend Yield**.")
        
        col_cols = ['Ticker', 'Preco', 'DY', 'PVP', 'MargemLiquida', 'Bazin_Preco_Teto']
        # Filtra apenas quem tem DY > 6
        df_bazin = df[df['DY'] >= 6].sort_values(by='DY', ascending=False).head(20)
        
        st.dataframe(
            df_bazin[col_cols].set_index('Ticker').style
            .format({'Preco': 'R$ {:.2f}', 'DY': '{:.2f}%', 'PVP': '{:.2f}', 'MargemLiquida': '{:.1f}%', 'Bazin_Preco_Teto': 'R$ {:.2f}'})
            .background_gradient(subset=['DY'], cmap='Greens'),
            use_container_width=True
        )

    # 2. ABA GRAHAM (Valor Intríseco)
    with tab2:
        st.subheader("O Método Graham: Preço Justo")
        st.markdown("Filtra empresas descontadas em relação ao seu lucro e patrimônio.")
        
        col_cols = ['Ticker', 'Preco', 'Graham_Preco_Justo', 'Graham_Potencial', 'PL', 'PVP']
        # Potencial > 0 e menor que 500 (para tirar distorções absurdas)
        df_graham = df[(df['Graham_Potencial'] > 0) & (df['Graham_Potencial'] < 500)].sort_values(by='Graham_Potencial', ascending=False).head(20)
        
        st.dataframe(
            df_graham[col_cols].set_index('Ticker').style
            .format({'Preco': 'R$ {:.2f}', 'Graham_Preco_Justo': 'R$ {:.2f}', 'Graham_Potencial': '{:.1f}%', 'PL': '{:.2f}', 'PVP': '{:.2f}'})
            .bar(subset=['Graham_Potencial'], color='#90ee90'),
            use_container_width=True
        )

    # 3. ABA MAGIC FORMULA (Qualidade + Preço)
    with tab3:
        st.subheader("A Fórmula Mágica (Joel Greenblatt)")
        st.markdown("Combina **Bom Preço** (Baixo EV/EBIT) com **Boa Gestão** (Alto ROIC).")
        
        col_cols = ['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']
        df_magic = df.dropna(subset=['Score_Magic']).sort_values(by='Score_Magic', ascending=True).head(20)
        
        st.dataframe(
            df_magic[col_cols].set_index('Ticker').style
            .format({'Preco': 'R$ {:.2f}', 'EV_EBIT': '{:.2f}', 'ROIC': '{:.2f}%', 'Score_Magic': '{:.0f}'})
            .background_gradient(subset=['ROIC'], cmap='Blues'),
            use_container_width=True
        )

    # 4. ABA CRESCIMENTO (Quality)
    with tab4:
        st.subheader("Empresas de Alta Qualidade (Quality Stocks)")
        st.markdown("Empresas com alto **ROE**, alta **Margem** e baixo endividamento.")
        
        # Cria um Score de Qualidade simples
        # Filtro: ROE > 15, Margem > 10, Dívida Controlada
        df_quality = df[
            (df['ROE'] > 15) & 
            (df['MargemLiquida'] > 10) & 
            (df['Divida_Patrimonio'] < 1.0)
        ].copy()
        
        df_quality = df_quality.sort_values(by='ROE', ascending=False).head(20)
        
        qual_cols = ['Ticker', 'Preco', 'ROE', 'MargemLiquida', 'Divida_Patrimonio', 'Crescimento5a']
        # Verifica colunas existentes para não dar erro
        qual_cols = [c for c in qual_cols if c in df_quality.columns]
        
        st.dataframe(
            df_quality[qual_cols].set_index('Ticker').style
            .format({'Preco': 'R$ {:.2f}', 'ROE': '{:.1f}%', 'MargemLiquida': '{:.1f}%', 'Divida_Patrimonio': '{:.2f}'})
            .background_gradient(subset=['ROE'], cmap='Purples'),
            use_container_width=True
        )

    # 5. ABA GERAL (Visualização Gráfica)
    with tab5:
        st.subheader("Mapa do Mercado")
        
        # Gráfico de Dispersão: Risco x Retorno
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Gráfico: Preço/Lucro vs ROE** (Onde estão as empresas baratas e rentáveis?)")
            
            # Remove outliers para o gráfico ficar bonito
            df_chart = df[
                (df['PL'] > 0) & (df['PL'] < 50) & 
                (df['ROE'] > 0) & (df['ROE'] < 50)
            ]
            
            fig = px.scatter(
                df_chart, 
                x='PL', 
                y='ROE', 
                hover_data=['Ticker', 'Preco', 'DY'],
                color='DY',
                size='Liquidez',
                title='Mapa de Oportunidades (Bola maior = Mais Líquida)',
                labels={'PL': 'P/L (Anos para retorno)', 'ROE': 'ROE (Rentabilidade)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.info("💡 **Como ler este gráfico?**")
            st.markdown("""
            - **Topo Esquerdo:** O "Filé Mignon". Empresas com ROE alto (rentáveis) e P/L baixo (baratas).
            - **Base Direita:** O "Mico". Empresas caras e pouco rentáveis.
            - **Cores:** Quanto mais amarelo, maior o Dividendo.
            """)

else:
    st.error("Falha crítica na conexão com a B3. Tente recarregar a página.")
