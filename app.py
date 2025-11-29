import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Titan", layout="wide")
st.title("⚡ Investidor Pro: Titan Edition")
st.markdown("### A plataforma completa: Sem erros, mais dados.")

# --- MOTOR DE DADOS ROBUSTO ---
@st.cache_data(ttl=300)
def carregar_dados_titan():
    try:
        # 1. Busca dados brutos
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # 2. Limpeza de Nomes de Colunas (Padronização)
        df.columns = [c.replace('.', '').replace('/', '').replace(' ', '').lower() for c in df.columns]
        
        # 3. Mapeamento Estendido (Mais colunas = Mais informação)
        mapa = {
            'papel': 'Ticker', 'ticker': 'Ticker', 'cotacao': 'Preco',
            'pl': 'PL', 'pvp': 'PVP', 'dy': 'DY', 'divyield': 'DY',
            'roe': 'ROE', 'roic': 'ROIC', 'evebit': 'EV_EBIT',
            'liq2meses': 'Liquidez', 'liq2m': 'Liquidez',
            'mrglíq': 'MargemLiquida', 'mrgliq': 'MargemLiquida',
            'divbrutpatr': 'Div_Patrimonio', 'liqcorr': 'Liq_Corrente',
            'crescrec5a': 'Cresc_5a'
        }
        df = df.rename(columns=mapa)
        
        # 4. Garantia Numérica (Evita qualquer erro de texto)
        cols_num = ['Preco', 'PL', 'PVP', 'DY', 'ROE', 'ROIC', 'EV_EBIT', 
                   'Liquidez', 'MargemLiquida', 'Div_Patrimonio', 'Liq_Corrente']
        
        for col in cols_num:
            if col not in df.columns: df[col] = 0.0
            # Coerce transforma erros em NaN, depois preenchemos com 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        # 5. Ajustes Percentuais (Heurística: se média < 1, multiplica por 100)
        if df['DY'].mean() < 1: df['DY'] *= 100
        if df['ROE'].mean() < 1: df['ROE'] *= 100
        if df['ROIC'].mean() < 1: df['ROIC'] *= 100
        if df['MargemLiquida'].mean() < 1: df['MargemLiquida'] *= 100
        
        return df
    except Exception as e:
        st.error(f"Erro no motor de dados: {e}")
        return pd.DataFrame()

# --- CÉREBRO DE CÁLCULOS ---
def calcular_indicadores(df):
    # GRAHAM
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    
    mask_pos = (df['LPA'] > 0) & (df['VPA'] > 0)
    df.loc[mask_pos, 'Graham_Preco'] = np.sqrt(22.5 * df.loc[mask_pos, 'LPA'] * df.loc[mask_pos, 'VPA'])
    df['Graham_Preco'] = df['Graham_Preco'].fillna(0)
    
    df['Graham_Upside'] = np.where(
        (df['Graham_Preco'] > 0) & (df['Preco'] > 0),
        ((df['Graham_Preco'] - df['Preco']) / df['Preco']) * 100,
        -999
    )

    # MAGIC FORMULA
    # Filtra apenas empresas operacionais válidas para o ranking
    df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
    if not df_magic.empty:
        df_magic['Rank_EV'] = df_magic['EV_EBIT'].rank(ascending=True)
        df_magic['Rank_ROIC'] = df_magic['ROIC'].rank(ascending=False)
        df_magic['Score_Magic'] = df_magic['Rank_EV'] + df_magic['Rank_ROIC']
        df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
    else:
        df['Score_Magic'] = 99999

    # BAZIN (Teto 6%)
    df['Bazin_Teto'] = np.where(df['DY'] > 0, df['Preco'] * (df['DY'] / 6), 0)
    
    return df

# --- FRONT-END ---
with st.spinner('Processando Big Data do Mercado...'):
    df_raw = carregar_dados_titan()

if not df_raw.empty:
    df = calcular_indicadores(df_raw)
    
    # --- FILTROS LATERAIS ---
    st.sidebar.header("🔍 Filtros Avançados")
    
    # Filtro de Liquidez (Padrão baixo para mostrar mais ações)
    liq_min = st.sidebar.select_slider(
        "Liquidez Diária Mínima:",
        options=[0, 1000, 50000, 200000, 1000000, 10000000],
        value=1000 # Começa baixo para mostrar tudo
    )
    
    df_view = df[df['Liquidez'] >= liq_min].copy()
    
    # Quantidade de ações
    st.sidebar.metric("Ações Encontradas", len(df_view))
    st.sidebar.markdown("---")
    st.sidebar.caption("Dados: Fundamentus | Engine: Titan v9.0")

    # --- ABAS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Top Dividendos", 
        "💎 Valor (Graham)", 
        "✨ Fórmula Mágica", 
        "📊 Dados Completos",
        "📈 Gráficos"
    ])

    # 1. DIVIDENDOS (NATIVE UI - SEM ERROS)
    with tab1:
        st.subheader("Maiores Pagadoras de Proventos")
        st.caption("Filtro: Dividend Yield acima de 0%")
        
        df_div = df_view[df_view['DY'] > 0].nlargest(50, 'DY')
        
        st.dataframe(
            df_div[['Ticker', 'Preco', 'DY', 'PVP', 'MargemLiquida', 'Bazin_Teto']],
            column_config={
                "Ticker": st.column_config.TextColumn("Ativo", width="small"),
                "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
                "DY": st.column_config.ProgressColumn("Dividend Yield", format="%.2f%%", min_value=0, max_value=20),
                "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
                "MargemLiquida": st.column_config.NumberColumn("Margem Líq.", format="%.1f%%"),
                "Bazin_Teto": st.column_config.NumberColumn("Preço Teto (Bazin)", format="R$ %.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )

    # 2. GRAHAM
    with tab2:
        st.subheader("As Mais Descontadas (Graham)")
        st.caption("Filtro: Potencial de Valorização Positivo (exclui prejuízo)")
        
        # Filtros de segurança para Graham
        df_graham = df_view[
            (df_view['Graham_Upside'] > 0) & 
            (df_view['Graham_Upside'] < 500) & # Tira distorções gigantes
            (df_view['Liquidez'] > 50000)      # Tira micos muito sem liquidez
        ].nlargest(50, 'Graham_Upside')
        
        st.dataframe(
            df_graham[['Ticker', 'Preco', 'Graham_Preco', 'Graham_Upside', 'PL', 'PVP']],
            column_config={
                "Ticker": st.column_config.TextColumn("Ativo", width="small"),
                "Preco": st.column_config.NumberColumn("Cotação", format="R$ %.2f"),
                "Graham_Preco": st.column_config.NumberColumn("Valor Justo", format="R$ %.2f"),
                "Graham_Upside": st.column_config.ProgressColumn("Potencial", format="%.1f%%", min_value=0, max_value=100),
                "PL": st.column_config.NumberColumn("P/L", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )

    # 3. MAGIC FORMULA
    with tab3:
        st.subheader("Qualidade + Preço (Greenblatt)")
        st.caption("Menor Score = Melhor Classificação")
        
        df_magic_view = df_view.nsmallest(50, 'Score_Magic')
        
        st.dataframe(
            df_magic_view[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']],
            column_config={
                "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
                "ROIC": st.column_config.ProgressColumn("ROIC (Qualidade)", format="%.1f%%", min_value=0, max_value=40),
                "EV_EBIT": st.column_config.NumberColumn("EV/EBIT (Preço)", format="%.2f"),
                "Score_Magic": st.column_config.NumberColumn("Score Final"),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )

    # 4. TABELA GERAL (MUITOS DADOS)
    with tab4:
        st.subheader("Screener Completo")
        st.markdown("Use a lupa no canto superior direito da tabela para pesquisar.")
        
        # Seleção de colunas ricas
        cols_full = ['Ticker', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Liquidez']
        
        st.dataframe(
            df_view[cols_full],
            column_config={
                "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
                "Liquidez": st.column_config.NumberColumn("Liquidez", format="R$ %.0f"),
                "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%"),
                "Div_Patrimonio": st.column_config.NumberColumn("Dívida/PL", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=700
        )

    # 5. GRÁFICOS
    with tab5:
        st.subheader("Análise Visual")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Gráfico Bolhas
            fig = px.scatter(
                df_view[df_view['PL'] < 50], 
                x='PL', y='ROE', 
                size='Liquidez', 
                color='DY',
                hover_name='Ticker',
                title="Risco x Retorno (Tamanho = Liquidez, Cor = Dividendos)",
                labels={'PL': 'Anos para Retorno (P/L)', 'ROE': 'Rentabilidade (ROE)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.info("🎯 **Onde investir?**")
            st.markdown("""
            Procure bolhas no **canto superior esquerdo**:
            
            * **ROE Alto:** Empresa rentável.
            * **P/L Baixo:** Empresa barata.
            * **Cor Amarela:** Paga bons dividendos.
            """)

else:
    st.error("Falha ao carregar dados. Tente recarregar a página (F5).")
