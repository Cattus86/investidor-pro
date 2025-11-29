import streamlit as st
import pandas as pd
import fundamentus
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Quant", layout="wide")
st.title("⚡ Investidor Pro: Plataforma Quantitativa")
st.markdown("Análise fundamentalista automatizada de todos os ativos da B3.")

# --- BARRA LATERAL (FILTROS GLOBAIS) ---
st.sidebar.header("🔍 Filtros Globais")
min_liquidez = st.sidebar.number_input("Liquidez Diária Mínima (R$):", value=200000, step=50000)

# --- MOTOR DE DADOS E CÁLCULOS ---
@st.cache_data(ttl=3600)
def carregar_base_completa():
    try:
        # 1. Baixar dados brutos do Fundamentus
        df = fundamentus.get_resultado()
        df = df.reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # 2. Limpeza Inicial
        # Converter colunas percentuais (que vêm como 0.15 para 15.0)
        cols_percent = ['Div.Yield', 'ROE', 'ROIC', 'Mrg. Líq.', 'Mrg. Ebit']
        for col in cols_percent:
            if col in df.columns:
                df[col] = df[col] * 100

        # Renomear colunas para ficar amigável
        mapa_colunas = {
            'Cotação': 'Preço',
            'Liq.2meses': 'Liquidez',
            'EV/EBIT': 'EV_EBIT'
        }
        df = df.rename(columns=mapa_colunas)
        
        # 3. Engenharia de Dados (Cálculos Derivados)
        
        # --- CÁLCULO DE GRAHAM ---
        # Graham precisa de LPA (Lucro por Ação) e VPA (Valor Patrimonial por Ação)
        # Como o fundamentus dá P/L e P/VP, vamos reverter a matemática:
        # LPA = Preço / PL
        # VPA = Preço / PVP
        
        df['LPA'] = np.where(df['P/L'] != 0, df['Preço'] / df['P/L'], 0)
        df['VPA'] = np.where(df['P/VP'] != 0, df['Preço'] / df['P/VP'], 0)
        
        def calcular_graham(row):
            if row['LPA'] > 0 and row['VPA'] > 0:
                return np.sqrt(22.5 * row['LPA'] * row['VPA'])
            return 0
            
        df['Preço Justo Graham'] = df.apply(calcular_graham, axis=1)
        df['Potencial Graham (%)'] = np.where(
            (df['Preço Justo Graham'] > 0) & (df['Preço'] > 0),
            ((df['Preço Justo Graham'] - df['Preço']) / df['Preço']) * 100,
            -999 # Valor baixo para ficar no fim da fila
        )

        # --- CÁLCULO MAGIC FORMULA (Greenblatt) ---
        # 1. Ranking de EV/EBIT (Menor é melhor) -> Barato
        # 2. Ranking de ROIC (Maior é melhor) -> Qualidade
        
        # Filtra apenas empresas com dados válidos para Magic Formula
        df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
        
        df_magic['Rank_EV_EBIT'] = df_magic['EV_EBIT'].rank(ascending=True)
        df_magic['Rank_ROIC'] = df_magic['ROIC'].rank(ascending=False)
        df_magic['Score_Magic'] = df_magic['Rank_EV_EBIT'] + df_magic['Rank_ROIC']
        
        # Traz o Score de volta para o dataframe principal
        df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')

        return df

    except Exception as e:
        st.error(f"Erro crítico ao processar dados: {e}")
        return pd.DataFrame()

# --- CARREGAMENTO ---
with st.spinner('Baixando e processando todos os ativos da B3...'):
    df_raw = carregar_base_completa()

if not df_raw.empty:
    # Aplica Filtro de Liquidez Global
    df = df_raw[df_raw['Liquidez'] >= min_liquidez].copy()
    
    st.success(f"Base carregada com sucesso! {len(df)} ativos analisados após filtro de liquidez.")

    # --- INTERFACE DE ABAS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "💰 Dividendos", "⚖️ Graham (Valor)", "✨ Fórmula Mágica"])

    # --- ABA 1: VISÃO GERAL (Todos os indicadores) ---
    with tab1:
        st.subheader("Screener Completo")
        st.write("Explore todos os indicadores fundamentalistas.")
        
        # Seleção de colunas para não ficar gigante
        cols_padrao = ['Ticker', 'Preço', 'P/L', 'P/VP', 'Div.Yield', 'ROE', 'Liquidez', 'Dív.Brut/ Patr.']
        all_cols = df.columns.tolist()
        cols_visiveis = st.multiselect("Colunas Visíveis:", all_cols, default=cols_padrao)
        
        st.dataframe(df[cols_visiveis].set_index('Ticker'), use_container_width=True, height=600)

    # --- ABA 2: DIVIDENDOS (Ranking) ---
    with tab2:
        st.subheader("🏆 Top Pagadoras de Dividendos")
        st.caption("Empresas ordenadas pelo Dividend Yield dos últimos 12 meses.")
        
        df_div = df.sort_values(by='Div.Yield', ascending=False).head(20)
        
        st.dataframe(
            df_div[['Ticker', 'Preço', 'Div.Yield', 'P/VP', 'Liquidez']].style
            .format({'Preço': 'R$ {:.2f}', 'Div.Yield': '{:.2f}%', 'P/VP': '{:.2f}'})
            .background_gradient(subset=['Div.Yield'], cmap='Greens'),
            use_container_width=True
        )

    # --- ABA 3: GRAHAM (Valuation Clássico) ---
    with tab3:
        st.subheader("💎 Oportunidades Segundo Benjamin Graham")
        st.markdown(r"Filtro baseado na fórmula: $V = \sqrt{22.5 \times LPA \times VPA}$")
        st.caption("Mostrando apenas ativos com Potencial positivo (> 0%). Cuidado com 'Bull Traps' (empresas quebradas).")
        
        # Filtra apenas quem tem margem positiva
        df_graham = df[df['Potencial Graham (%)'] > 0].sort_values(by='Potencial Graham (%)', ascending=False)
        
        st.dataframe(
            df_graham[['Ticker', 'Preço', 'Preço Justo Graham', 'Potencial Graham (%)', 'P/L', 'P/VP']].head(30).style
            .format({'Preço': 'R$ {:.2f}', 'Preço Justo Graham': 'R$ {:.2f}', 'Potencial Graham (%)': '{:.2f}%'})
            .bar(subset=['Potencial Graham (%)'], color='lightgreen'),
            use_container_width=True
        )

    # --- ABA 4: FÓRMULA MÁGICA (Greenblatt) ---
    with tab4:
        st.subheader("✨ Ranking da Fórmula Mágica")
        st.markdown("**Estratégia:** Comprar empresas *boas* (Alto ROIC) a preços *baratos* (Baixo EV/EBIT).")
        st.caption("Quanto menor o 'Score Magic', melhor a classificação.")
        
        # Filtra nulos e ordena pelo Score (Menor é melhor)
        df_magic_view = df.dropna(subset=['Score_Magic']).sort_values(by='Score_Magic', ascending=True).head(30)
        
        st.dataframe(
            df_magic_view[['Ticker', 'Preço', 'EV_EBIT', 'ROIC', 'Score_Magic']].style
            .format({'Preço': 'R$ {:.2f}', 'EV_EBIT': '{:.2f}', 'ROIC': '{:.2f}%', 'Score_Magic': '{:.0f}'})
            .background_gradient(subset=['Score_Magic'], cmap='Blues_r'), # Invertido: azul escuro para os primeiros
            use_container_width=True
        )

else:
    st.warning("Aguardando carregamento dos dados...")
