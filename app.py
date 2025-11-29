import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Titanium", layout="wide")
st.title("💎 Investidor Pro: Titanium Edition")
st.markdown("### A Plataforma Definitiva: Sem erros, Análise Automática e Rankings.")

# --- 2. FUNÇÕES DE LIMPEZA E DADOS ---
def limpar_numero_ptbr(valor):
    """Converte números brasileiros (texto) para float do Python."""
    if isinstance(valor, str):
        valor_limpo = valor.replace('.', '').replace(',', '.').replace('%', '').strip()
        try:
            return float(valor_limpo)
        except:
            return 0.0
    return float(valor) if valor else 0.0

@st.cache_data(ttl=300)
def carregar_dados_titanium():
    try:
        # Baixa dados brutos
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Renomeação Segura (Linha a linha para não quebrar)
        mapa = {
            'Cotação': 'Preco',
            'P/L': 'PL',
            'P/VP': 'PVP',
            'Div.Yield': 'DY',
            'ROE': 'ROE',
            'ROIC': 'ROIC',
            'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez',
            'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio',
            'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        # Filtra apenas o que existe para evitar erros
        colunas_finais = ['Ticker'] + [c for c in mapa.keys() if c in df.columns]
        df = df[colunas_finais].copy()
        df.rename(columns=mapa, inplace=True)
        
        # Limpeza Numérica
        for col in df.columns:
            if col != 'Ticker':
                df[col] = df[col].apply(limpar_numero_ptbr)
        
        # Ajustes de Escala
        if 'DY' in df.columns and df['DY'].mean() < 1: 
            df['DY'] = df['DY'] * 100
        if 'ROE' in df.columns and df['ROE'].mean() < 1: 
            df['ROE'] = df['ROE'] * 100
        if 'MargemLiquida' in df.columns and df['MargemLiquida'].mean() < 1: 
            df['MargemLiquida'] = df['MargemLiquida'] * 100
        
        return df
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    # Graham (Cálculo quebrado em etapas para não dar erro de sintaxe)
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    
    mask_valida = (df['LPA'] > 0) & (df['VPA'] > 0)
    df.loc[mask_valida, 'Graham_Valor'] = np.sqrt(22.5 * df.loc[mask_valida, 'LPA'] * df.loc[mask_valida, 'VPA'])
    df['Graham_Valor'] = df['Graham_Valor'].fillna(0)
    
    # Upside (Cálculo Blindado)
    diferenca = df['Graham_Valor'] - df['Preco']
    df['Graham_Upside'] = np.where(
        (df['Graham_Valor'] > 0) & (df['Preco'] > 0),
        (diferenca / df['Preco']) * 100,
        -999
    )

    # Magic Formula
    df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
    if not df_magic.empty:
        df_magic['R_EV'] = df_magic['EV_EBIT'].rank(ascending=True)
        df_magic['R_ROIC'] = df_magic['ROIC'].rank(ascending=False)
        df_magic['Score_Magic'] = df_magic['R_EV'] + df_magic['R_ROIC']
        df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
    else:
        df['Score_Magic'] = 99999
        
    # Bazin
    df['Bazin_Teto'] = np.where(df['DY'] > 0, df['Preco'] * (df['DY'] / 6), 0)
    
    return df

# --- 3. ROBÔ ANALISTA ---
def analisar_ativo(row):
    score = 5
    texto = []
    
    # Dividendos
    if row['DY'] > 12:
        texto.append("🟢 **Dividendos:** Yield altíssimo (>12%). Cuidado com a sustentabilidade.")
        score += 2
    elif row['DY'] > 6:
        texto.append("🔵 **Dividendos:** Bom pagador (>6%).")
        score += 1
    else:
        texto.append("⚪ **Dividendos:** Baixo yield.")
        score -= 1
        
    # Preço
    if row['PL'] <= 0:
        texto.append("🔴 **Lucro:** Prejuízo recente.")
        score -= 3
    elif row['PL'] < 5:
        texto.append("🟢 **Preço:** Muito descontada (P/L < 5).")
        score += 2
    elif row['PL'] < 15:
        texto.append("🔵 **Preço:** Justo.")
        score += 1
        
    # Qualidade
    if row['ROE'] > 20:
        texto.append("🔥 **Qualidade:** Alta rentabilidade (ROE > 20%).")
        score += 2
    elif row['ROE'] < 5:
        texto.append("❄️ **Qualidade:** Rentabilidade baixa.")
        score -= 1

    # Graham
    if row['Graham_Upside'] > 20:
        texto.append(f"💎 **Graham:** Potencial de +{row['Graham_Upside']:.0f}%.")
        score += 1
        
    return score, "\n\n".join(texto)

# --- 4. INTERFACE ---
with st.spinner('Conectando à B3...'):
    df_raw = carregar_dados_titanium()

if not df_raw.empty:
    df = calcular_indicadores(df_raw)
    
    # Filtros
    st.sidebar.header("🔍 Filtros")
    busca = st.sidebar.text_input("Buscar (ex: PETR4):").upper().strip()
    liq_min = st.sidebar.select_slider("Liquidez Mínima", options=[0, 50000, 200000, 1000000], value=50000)
    
    df_view = df[df['Liquidez'] >= liq_min].copy()
    if busca:
        df_view = df_view[df_view['Ticker'].str.contains(busca)]
        
    st.sidebar.download_button("📥 Baixar CSV", df_view.to_csv(sep=';', decimal=','), "acoes.csv", "text/csv")

    # Área do Analista
    col_kpi, col_msg = st.columns([1, 2])
    if 'ativo_selecionado' not in st.session_state:
        st.session_state['ativo_selecionado'] = None

    # Abas
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Dividendos", "💎 Graham", "✨ Magic Formula", "📊 Gráfico"])
    
    # Configuração Visual da Tabela
    cfg = {
        "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
        "DY": st.column_config.ProgressColumn("Yield", format="%.2f%%", min_value=0, max_value=15),
        "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
        "PL": st.column_config.NumberColumn("P/L", format="%.2f"),
        "Graham_Upside": st.column_config.ProgressColumn("Upside", format="%.0f%%", min_value=0, max_value=100),
        "Bazin_Teto": st.column_config.NumberColumn("Teto Bazin", format="R$ %.2f"),
        "Score_Magic": st.column_config.NumberColumn("Score", format="%d"),
        "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%")
    }

    # Função de Clique
    def on_click_tabela(evento, dataframe_origem):
        if len(evento.selection.rows) > 0:
            idx = evento.selection.rows[0]
            st.session_state['ativo_selecionado'] = dataframe_origem.iloc[idx]

    with tab1:
        st.subheader("Top Dividendos")
        df_t1 = df_view.nlargest(100, 'DY')
        ev1 = st.dataframe(
            df_t1[['Ticker', 'Preco', 'DY', 'PVP', 'Bazin_Teto']], 
            column_config=cfg, hide_index=True, use_container_width=True, 
            on_select="rerun", selection_mode="single-row"
        )
        on_click_tabela(ev1, df_t1)

    with tab2:
        st.subheader("Top Graham")
        df_t2 = df_view[(df_view['Graham_Upside'] > 0) & (df_view['Graham_Upside'] < 500)].nlargest(100, 'Graham_Upside')
        ev2 = st.dataframe(
            df_t2[['Ticker', 'Preco', 'Graham_Valor', 'Graham_Upside', 'PL']], 
            column_config={
                "Graham_Valor": st.column_config.NumberColumn("Valor Justo", format="R$ %.2f"),
                **cfg
            }, hide_index=True, use_container_width=True, 
            on_select="rerun", selection_mode="single-row"
        )
        on_click_tabela(ev2, df_t2)

    with tab3:
        st.subheader("Magic Formula")
        df_t3 = df_view.nsmallest(100, 'Score_Magic')
        ev3 = st.dataframe(
            df_t3[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']], 
            column_config=cfg, hide_index=True, use_container_width=True, 
            on_select="rerun", selection_mode="single-row"
        )
        on_click_tabela(ev3, df_t3)

    with tab4:
        st.subheader("Mapa de Calor")
        df_chart = df_view[(df_view['PL'] > 0) & (df_view['PL'] < 50) & (df_view['ROE'] > 0) & (df_view['ROE'] < 60)]
        fig = px.scatter(
            df_chart, x='PL', y='ROE', size='Liquidez', color='DY', 
            hover_name='Ticker', text='Ticker', title="Risco x Retorno", 
            color_continuous_scale='RdYlGn'
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

    # Mostrador do Analista
    if st.session_state['ativo_selecionado'] is not None:
        row = st.session_state['ativo_selecionado']
        score, texto = analisar_ativo(row)
        with col_kpi:
            st.metric("Nota", f"{score}/10")
            st.info(row['Ticker'])
        with col_msg:
            st.success(texto)

else:
    st.error("Erro ao carregar dados. Recarregue a página.")
