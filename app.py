import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Investidor Pro | Platinum", layout="wide")
st.title("💎 Investidor Pro: Platinum Edition")

# --- O SEGREDO (LAVADORA DE DADOS PT-BR) ---
def limpar_numero_ptbr(valor):
    """
    Força bruta para consertar números brasileiros.
    Aceita texto '1.200,50' e devolve float 1200.50
    Aceita float 10.5 e devolve float 10.5
    """
    if isinstance(valor, str):
        # Remove pontos de milhar e troca vírgula por ponto
        valor_limpo = valor.replace('.', '').replace(',', '.').replace('%', '').strip()
        try:
            return float(valor_limpo)
        except:
            return 0.0
    return float(valor) if valor else 0.0

# --- MOTOR DE DADOS ---
@st.cache_data(ttl=300)
def carregar_dados_platinum():
    try:
        # 1. Baixar dados brutos (Raw = menos formatação, mais seguro)
        df = fundamentus.get_resultado_raw().reset_index()
        
        # 2. Renomear colunas para facilitar nossa vida
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Mapa manual das colunas que importam (Nome Original -> Nosso Nome)
        # O Fundamentus Raw costuma ter estes nomes:
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
            'Dív.Brut/ Patr.': 'Div_Patrimonio'
        }
        
        # Filtra e renomeia apenas o que existe
        colunas_uteis = [c for c in mapa_colunas.keys() if c in df.columns]
        df = df[['Ticker'] + colunas_uteis].copy()
        df.rename(columns=mapa_colunas, inplace=True)
        
        # 3. APLICAÇÃO DA LIMPEZA (A CORREÇÃO DO ZERO)
        cols_numericas = ['Preco', 'PL', 'PVP', 'DY', 'ROE', 'ROIC', 'EV_EBIT', 'Liquidez', 'MargemLiquida', 'Div_Patrimonio']
        
        for col in cols_numericas:
            if col in df.columns:
                # Aplica a função linha a linha
                df[col] = df[col].apply(limpar_numero_ptbr)
        
        # 4. Ajustes de Escala (Percentuais)
        # Se o DY médio for menor que 1 (ex: 0.06), multiplica por 100 para virar 6%
        if 'DY' in df.columns and df['DY'].mean() < 1: df['DY'] *= 100
        if 'ROE' in df.columns and df['ROE'].mean() < 1: df['ROE'] *= 100
        
        return df

    except Exception as e:
        st.error(f"Erro crítico no motor de dados: {e}")
        return pd.DataFrame()

# --- CÁLCULOS FINANCEIROS ---
def calcular_kpis(df):
    # GRAHAM (Agora vai funcionar porque Preço > 0)
    # LPA = Preço / PL
    # VPA = Preço / PVP
    
    # Evita divisão por zero
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    
    # Graham exige Lucro e Patrimônio Positivos
    mask_valida = (df['LPA'] > 0) & (df['VPA'] > 0)
    
    df.loc[mask_valida, 'Graham_Valor'] = np.sqrt(22.5 * df.loc[mask_valida, 'LPA'] * df.loc[mask_valida, 'VPA'])
    df['Graham_Valor'] = df['Graham_Valor'].fillna(0)
    
    df['Graham_Potencial'] = np.where(
        (df['Graham_Valor'] > 0) & (df['Preco'] > 0),
        ((df['Graham_Valor'] - df['Preco']) / df['Preco']) * 100,
        -999
    )

    # MAGIC FORMULA
    # Filtra empresas saudáveis
    df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
    if not df_magic.empty:
        df_magic['R_EV'] = df_magic['EV_EBIT'].rank(ascending=True)
        df_magic['R_ROIC'] = df_magic['ROIC'].rank(ascending=False)
        df_magic['Score_Magic'] = df_magic['R_EV'] + df_magic['R_ROIC']
        df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
    else:
        df['Score_Magic'] = 99999
        
    return df

# --- INTERFACE ---
with st.spinner('Carregando dados da B3...'):
    df_raw = carregar_dados_platinum()

if not df_raw.empty:
    df = calcular_kpis(df_raw)
    
    # --- BARRA LATERAL ---
    st.sidebar.header("🔍 Filtros de Mercado")
    
    # Filtro de Liquidez INTELIGENTE
    # Vamos filtrar ações "Mico" (Liquidez < 50k) por padrão para limpar o gráfico
    liq_min = st.sidebar.number_input("Liquidez Diária Mínima (R$)", value=50000.0, step=10000.0)
    df_view = df[df['Liquidez'] >= liq_min].copy()
    
    st.sidebar.markdown(f"**Ativos Analisados:** {len(df_view)}")
    
    # --- ABAS ---
    tab1, tab2, tab3 = st.tabs(["💰 Ranking Dividendos", "⚖️ Ranking Graham", "📊 Mapa do Mercado"])
    
    # 1. DIVIDENDOS
    with tab1:
        st.subheader("Top Pagadoras (Yield)")
        
        # Remove bizarrices (Yield > 100% geralmente é erro de dado ou evento não recorrente)
        df_div = df_view[(df_view['DY'] > 0) & (df_view['DY'] < 100)].nlargest(30, 'DY')
        
        st.dataframe(
            df_div[['Ticker', 'Preco', 'DY', 'PVP', 'MargemLiquida']],
            column_config={
                "Preco": st.column_config.NumberColumn("Preço Atual", format="R$ %.2f"),
                "DY": st.column_config.ProgressColumn("Dividend Yield", format="%.2f%%", min_value=0, max_value=20),
                "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
                "MargemLiquida": st.column_config.NumberColumn("Margem Líq.", format="%.1f%%"),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
    # 2. GRAHAM
    with tab2:
        st.subheader("Oportunidades de Valor (Graham)")
        
        # Filtra apenas o que tem potencial positivo e real (tira distorções > 300%)
        df_graham = df_view[
            (df_view['Graham_Potencial'] > 0) & 
            (df_view['Graham_Potencial'] < 300)
        ].nlargest(30, 'Graham_Potencial')
        
        st.dataframe(
            df_graham[['Ticker', 'Preco', 'Graham_Valor', 'Graham_Potencial', 'PL']],
            column_config={
                "Preco": st.column_config.NumberColumn("Preço Tela", format="R$ %.2f"),
                "Graham_Valor": st.column_config.NumberColumn("Preço Justo", format="R$ %.2f"),
                "Graham_Potencial": st.column_config.ProgressColumn("Potencial", format="%.1f%%", min_value=0, max_value=100),
                "PL": st.column_config.NumberColumn("P/L", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
    # 3. GRÁFICO (REFORMULADO)
    with tab3:
        st.subheader("Mapa de Calor: Risco x Retorno")
        st.markdown("Eixo X: **P/L** (Quanto mais à esquerda, mais barato). Eixo Y: **ROE** (Quanto mais alto, melhor).")
        
        # Limpeza para o gráfico não ficar com escala "quebrada" por outliers
        df_chart = df_view[
            (df_view['PL'] > 0) & (df_view['PL'] < 40) & 
            (df_view['ROE'] > 0) & (df_view['ROE'] < 60)
        ]
        
        fig = px.scatter(
            df_chart,
            x='PL',
            y='ROE',
            size='Liquidez', # Bolha maior = Mais fácil de comprar/vender
            color='DY',      # Cor = Dividendos
            hover_name='Ticker',
            hover_data={'Preco': ':.2f', 'PVP': ':.2f'},
            text='Ticker',   # Mostra o nome da ação na bolha
            title="Onde estão as melhores empresas?",
            color_continuous_scale='RdYlGn', # Vermelho (pouco div) -> Verde (muito div)
            height=600
        )
        
        # Melhora o visual do texto nas bolhas
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Erro ao carregar dados. Tente atualizar a página.")
