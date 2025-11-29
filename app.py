import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURAÇÃO DA PÁGINA E CSS ---
st.set_page_config(page_title="Investidor Pro | Platinum", layout="wide", initial_sidebar_state="expanded")

# CSS Profissional
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #00c853;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 5px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00c853 !important;
        color: white !important;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #131722;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a2e39;
    }
</style>
""", unsafe_allow_html=True)

st.title("💎 Investidor Pro: Platinum Edition")
st.markdown("##### Plataforma de Análise Quantitativa Avançada")

# --- 2. MOTOR DE DADOS ---
def limpar_numero_ptbr(valor):
    if isinstance(valor, str):
        valor_limpo = valor.replace('.', '').replace(',', '.').replace('%', '').strip()
        try:
            return float(valor_limpo)
        except:
            return 0.0
    return float(valor) if valor else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def carregar_dados_platinum():
    try:
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        mapa = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio'
        }
        
        cols = ['Ticker'] + [c for c in mapa.keys() if c in df.columns]
        df = df[cols].copy()
        df.rename(columns=mapa, inplace=True)
        
        for col in df.columns:
            if col != 'Ticker':
                df[col] = df[col].apply(limpar_numero_ptbr)
                
        # Ajustes de percentual
        if 'DY' in df.columns and df['DY'].mean() < 1: 
            df['DY'] = df['DY'] * 100
        if 'ROE' in df.columns and df['ROE'].mean() < 1: 
            df['ROE'] = df['ROE'] * 100
        if 'MargemLiquida' in df.columns and df['MargemLiquida'].mean() < 1: 
            df['MargemLiquida'] = df['MargemLiquida'] * 100
            
        return df
    except Exception as e:
        st.error(f"Erro crítico: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    # Graham
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    
    mask_valida = (df['LPA'] > 0) & (df['VPA'] > 0)
    df.loc[mask_valida, 'Graham_Valor'] = np.sqrt(22.5 * df.loc[mask_valida, 'LPA'] * df.loc[mask_valida, 'VPA'])
    df['Graham_Valor'] = df['Graham_Valor'].fillna(0)
    
    df['Graham_Upside'] = np.where(
        (df['Graham_Valor'] > 0) & (df['Preco'] > 0),
        ((df['Graham_Valor'] - df['Preco']) / df['Preco']) * 100, 
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

# --- 3. ROBÔ ANALISTA (CORRIGIDO E EXPANDIDO) ---
def analisar_ativo(row):
    score = 5
    texto = []
    
    # Dividendos
    if row['DY'] > 10:
        texto.append("🟢 **Yield:** Altíssimo (>10%). Verificar recorrencia.")
        score += 2
    elif row['DY'] > 6:
        texto.append("🔵 **Yield:** Atrativo (>6%).")
        score += 1
    else:
        texto.append("⚪ **Yield:** Baixo.")
        score -= 1
        
    # Preço (P/L)
    if row['PL'] <= 0:
        texto.append("🔴 **P/L:** Empresa com prejuízo.")
        score -= 3
    elif row['PL'] < 5:
        texto.append("🟢 **P/L:** Muito descontada (<5).")
        score += 2
    elif row['PL'] < 15:
        texto.append("🔵 **P/L:** Razoável.")
        score += 1
        
    # Qualidade (ROE)
    if row['ROE'] > 15:
        texto.append("🔥 **ROE:** Alta rentabilidade (>15%).")
        score += 2
    elif row['ROE'] < 5:
        texto.append("❄️ **ROE:** Rentabilidade baixa.")
        score -= 1
        
    # Graham
    if row['Graham_Upside'] > 20:
        texto.append(f"💎 **Graham:** Potencial teórico de +{row['Graham_Upside']:.0f}%.")
        score += 1
    
    return score, "\n\n".join(texto)

# --- 4. INTERFACE ---
with st.spinner('Sincronizando dados do mercado...'):
    df_raw = carregar_dados_platinum()

if not df_raw.empty:
    df = calcular_indicadores(df_raw)
    
    # Sidebar
    st.sidebar.header("🔍 Filtros Avançados")
    busca = st.sidebar.text_input("Buscar Ticker:", placeholder="Ex: VALE3").upper().strip()
    liq_min = st.sidebar.select_slider("Liquidez Mínima:", options=[0, 50000, 200000, 1000000, 5000000], value=200000)
    
    df_view = df[df['Liquidez'] >= liq_min].copy()
    if busca:
        df_view = df_view[df_view['Ticker'].str.contains(busca)]
    
    st.sidebar.markdown("---")
    st.sidebar.download_button("📥 Baixar CSV", df_view.to_csv(sep=';', decimal=','), "relatorio_platinum.csv", "text/csv")

    # Dashboard Topo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ativos", len(df_view))
    
    media_dy = df_view[df_view['DY']>0]['DY'].mean()
    c2.metric("Média Yield", f"{media_dy:.2f}%")
    
    media_pl = df_view[(df_view['PL']>0) & (df_view['PL']<100)]['PL'].mean()
    c3.metric("Média P/L", f"{media_pl:.1f}x")
    
    liq_total = df_view['Liquidez'].sum()/1_000_000_000
    c4.metric("Liquidez Total", f"R$ {liq_total:.1f} Bi")
    
    st.divider()

    # Layout Principal
    col_main, col_details = st.columns([2.2, 1])

    if 'ativo_selecionado' not in st.session_state:
        st.session_state['ativo_selecionado'] = None

    def on_click_tabela(evento, dataframe_origem):
        if len(evento.selection.rows) > 0:
            idx = evento.selection.rows[0]
            st.session_state['ativo_selecionado'] = dataframe_origem.iloc[idx]

    # Config Colunas
    cfg = {
        "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
        "DY": st.column_config.ProgressColumn("Yield", format="%.2f%%", min_value=0, max_value=15),
        "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
        "PL": st.column_config.NumberColumn("P/L", format="%.2f"),
        "Graham_Upside": st.column_config.ProgressColumn("Potencial", format="%.0f%%", min_value=0, max_value=100),
        "Bazin_Teto": st.column_config.NumberColumn("Teto Bazin", format="R$ %.2f"),
        "Score_Magic": st.column_config.NumberColumn("Score", format="%d"),
        "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%")
    }

    with col_main:
        t1, t2, t3, t4 = st.tabs(["💰 Dividendos", "💎 Graham", "✨ Magic Formula", "📈 Gráficos"])
        
        with t1:
            df_t1 = df_view.nlargest(100, 'DY')
            st.caption("Selecione para ver detalhes.")
            ev1 = st.dataframe(df_t1[['Ticker', 'Preco', 'DY', 'PVP', 'Bazin_Teto', 'ROE']], 
                             column_config=cfg, hide_index=True, use_container_width=True, 
                             on_select="rerun", selection_mode="single-row", height=500)
            on_click_tabela(ev1, df_t1)

        with t2:
            df_t2 = df_view[(df_view['Graham_Upside'] > 0) & (df_view['Graham_Upside'] < 500)].nlargest(100, 'Graham_Upside')
            st.caption("Selecione para ver detalhes.")
            ev2 = st.dataframe(df_t2[['Ticker', 'Preco', 'Graham_Valor', 'Graham_Upside', 'PL', 'PVP']], 
                             column_config={"Graham_Valor": st.column_config.NumberColumn("Justo", format="R$ %.2f"), **cfg}, 
                             hide_index=True, use_container_width=True, 
                             on_select="rerun", selection_mode="single-row", height=500)
            on_click_tabela(ev2, df_t2)

        with t3:
            df_t3 = df_view.nsmallest(100, 'Score_Magic')
            st.caption("Selecione para ver detalhes.")
            ev3 = st.dataframe(df_t3[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']], 
                             column_config=cfg, hide_index=True, use_container_width=True, 
                             on_select="rerun", selection_mode="single-row", height=500)
            on_click_tabela(ev3, df_t3)

        with t4:
            st.subheader("Analytics")
            # Barras Top Yield
            top_dy = df_view.nlargest(15, 'DY').sort_values(by='DY', ascending=True)
            fig_bar = px.bar(top_dy, x='DY', y='Ticker', orientation='h', 
                             title="Top 15 Yields", text_auto='.2f', color='DY', color_continuous_scale='greens')
            st.plotly_chart(fig_bar, use_container_width=True)
            st.divider()
            
            # Scatter
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                df_sc = df_view[(df_view['PL'] > 0) & (df_view['PL'] < 50) & (df_view['ROE'] > 0) & (df_view['ROE'] < 60)]
                fig_sc = px.scatter(df_sc, x='PL', y='ROE', size='Liquidez', color='DY', 
                                    hover_name='Ticker', title="Risco x Retorno", color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_sc, use_container_width=True)
            
            with c_g2:
                df_h = df_view[(df_view['PL'] > -20) & (df_view['PL'] < 50)]
                fig_h = px.histogram(df_h, x="PL", nbins=30, title="Distribuição P/L", color_discrete_sequence=['#00c853'])
                st.plotly_chart(fig_h, use_container_width=True)

    # Painel Lateral
    with col_details:
        st.subheader("📊 Raio-X")
        if st.session_state['ativo_selecionado'] is not None:
            row = st.session_state['ativo_selecionado']
            score, texto = analisar_ativo(row)
            
            st.markdown(f"# {row['Ticker']}")
            st.metric("Preço", f"R$ {row['Preco']:.2f}")
            st.divider()
            
            c_s, c_t = st.columns([1, 3])
            with c_s:
                st.metric("Score", f"{score}/10")
            
            st.info(texto)
            st.divider()
            
            # Gauge Chart
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number+delta", value = row['DY'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Yield vs Média"},
                delta = {'reference': media_dy, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge = {'axis': {'range': [None, max(20, row['DY'] + 5)]}, 'bar': {'color': "#00c853"}}
            ))
            fig_g.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.info("👈 Clique em uma ação na tabela para ver a análise completa.")

else:
    st.error("Erro ao carregar dados.")
