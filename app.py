import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

# --- 1. CONFIGURAÇÃO VISUAL (DARK PRO) ---
st.set_page_config(page_title="Titanium Pro II", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Fundo e Fontes */
    .stApp { background-color: #0e1117; font-family: 'Roboto', sans-serif; }
    
    /* Métricas - Estilo Neon Clean */
    [data-testid="stMetricValue"] { font-size: 1.5rem; color: #00e676; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem; color: #b0b3b8; }
    
    /* Tabelas Profissionais */
    [data-testid="stDataFrame"] { border: 1px solid #2d3748; border-radius: 5px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #2d3748; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #1a202c; padding: 5px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { height: 40px; border: none; color: #a0aec0; }
    .stTabs [aria-selected="true"] { background-color: #2d3748 !important; color: #4fd1c5 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro II: Market Terminal")

# --- 2. TAXONOMIA DE SETORES (EXPANDIDA) ---
# Mapeamento manual dos principais ativos para garantir precisão
SETORES_B3 = {
    'Financeiro': ['ITUB4', 'BBDC4', 'BBAS3', 'SANB11', 'BPAC11', 'B3SA3', 'CIEL3', 'BBSE3', 'CXSE3', 'IRBR3', 'PSSA3', 'ITSA4', 'ABCB4', 'BRSR6'],
    'Utilidade Pública': ['ELET3', 'ELET6', 'EQTL3', 'CPLE6', 'CMIG4', 'TRPL4', 'TAEE11', 'EGIE3', 'NEOE3', 'LIGT3', 'SBSP3', 'CSMG3', 'SAPR11', 'ALUP11', 'AURE3'],
    'Materiais Básicos': ['VALE3', 'GGBR4', 'CSNA3', 'USIM5', 'GOAU4', 'SUZB3', 'KLBN11', 'BRKM5', 'CMIN3', 'FESA4', 'CBAV3', 'UNIP6', 'DXCO3'],
    'Petróleo & Gás': ['PETR4', 'PETR3', 'PRIO3', 'UGPA3', 'CSAN3', 'VBBR3', 'RRRP3', 'RECV3', 'ENAT3', 'BRAV3', 'RAIZ4'],
    'Consumo Cíclico': ['MGLU3', 'LREN3', 'ARZZ3', 'SOMA3', 'PETZ3', 'AMER3', 'BHIA3', 'ALPA4', 'CVCB3', 'RENT3', 'MOVI3', 'EZTC3', 'CYRE3', 'MRVE3', 'TEND3', 'DIRR3', 'CURY3'],
    'Consumo Não Cíclico': ['ABEV3', 'JBSS3', 'BRFS3', 'MRFG3', 'BEEF3', 'SMTO3', 'MDIA3', 'CAML3', 'CRFB3', 'ASAI3', 'NTCO3', 'AGRO3', 'SLCE3', 'TTEN3'],
    'Saúde': ['RADL3', 'RDOR3', 'HAPV3', 'FLRY3', 'QUAL3', 'ODPV3', 'MATD3', 'VVEO3'],
    'Tecnologia': ['WEGE3', 'TOTS3', 'LWSA3', 'POSI3', 'INTB3', 'MLAS3'],
    'Indústria & Logística': ['EMBR3', 'AZUL4', 'GOLL4', 'CCRO3', 'ECOR3', 'RAIL3', 'HBSA3', 'STBP3', 'TUPY3', 'POMO4', 'RAPT4', 'KEPL3'],
    'Imobiliário (Holdings)': ['MULT3', 'IGTI11', 'ALOS3', 'JHSF3', 'LOGG3', 'HBRE3']
}

def classificar_setor(ticker):
    t = ticker.upper().strip()
    for setor, lista in SETORES_B3.items():
        if t in lista: return setor
    return "Outros / Small Caps"

# --- 3. MOTOR DE DADOS ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_data():
    try:
        # 1. Fundamentus Raw
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Mapeamento Completo
        cols_map = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a',
            'Patrim. Líq': 'Patrimonio', 'Ativo': 'Ativos', 'Dív.Líquida/EBITDA': 'DL_EBITDA'
        }
        
        # Filtra colunas existentes
        valid_cols = ['Ticker'] + [c for c in cols_map.keys() if c in df.columns]
        df = df[valid_cols].copy().rename(columns=cols_map)
        
        # Limpeza Numérica
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(clean_float)
            
        # Ajustes Percentuais
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        # Classificação Setorial
        df['Setor'] = df['Ticker'].apply(classificar_setor)
        
        # 2. Cálculos Quant
        # Graham
        df['Graham_Fair'] = np.where((df['PL']>0)&(df['PVP']>0), np.sqrt(22.5 * (df['Preco']/df['PL']) * (df['Preco']/df['PVP'])), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        # Bazin
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco']*(df['DY']/6), 0)
        # Magic Formula (Score)
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Rank_EV'] = df_m['EV_EBIT'].rank(ascending=True)
            df_m['Rank_ROIC'] = df_m['ROIC'].rank(ascending=False)
            df_m['Score_Magic'] = df_m['Rank_EV'] + df_m['Rank_ROIC']
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999
            
        return df
    except: return pd.DataFrame()

# --- 4. INTERFACE E LÓGICA ---

# A. Sidebar
st.sidebar.header("🎛️ Filtros Avançados")
usar_yahoo = st.sidebar.checkbox("📡 Dados Técnicos (Yahoo)", value=True, help="Ativa gráficos e momentum")

with st.spinner('Carregando Engine...'):
    df_base = get_data()

if df_base.empty:
    st.error("Falha ao conectar na B3.")
    st.stop()

# Filtros
busca = st.sidebar.text_input("Ticker", placeholder="PETR4").upper()
setores_disp = ["Todos"] + sorted(list(set(SETORES_B3.keys()))) + ["Outros / Small Caps"]
setor_f = st.sidebar.selectbox("Setor", setores_disp)
# Correção do Slider (Valor incluso nas opções)
liq_f = st.sidebar.select_slider("Liquidez Mínima", options=[0, 100000, 200000, 1000000, 10000000], value=200000)

# Aplicação
df_view = df_base[df_base['Liquidez'] >= liq_f].copy()
if setor_f != "Todos": df_view = df_view[df_view['Setor'] == setor_f]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Cálculo Momentum Lote (apenas visualização tabela)
if usar_yahoo:
    with st.spinner("Calculando Tendências..."):
        # Top 50 por liquidez para não travar
        top_list = df_view.nlargest(50, 'Liquidez')['Ticker'].tolist()
        t_sa = [t+".SA" for t in top_list]
        try:
            h = yf.download(t_sa, period="6mo", progress=False)['Adj Close']
            if isinstance(h, pd.Series): h = h.to_frame(name=t_sa[0])
            res = {}
            for col in h.columns:
                s = h[col].dropna()
                if len(s)>10:
                    ret = ((s.iloc[-1] - s.iloc[0]) / s.iloc[0]) * 100
                    res[col.replace('.SA','')] = ret
            df_view['Momentum'] = df_view['Ticker'].map(res).fillna(0)
            
            # Métrica de Breadth (Amplitude)
            altas = len([k for k,v in res.items() if v > 0])
            baixas = len(res) - altas
        except:
            df_view['Momentum'] = 0
            altas, baixas = 0, 0
else:
    df_view['Momentum'] = 0
    altas, baixas = 0, 0

# B. Dashboard Topo
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ativos Filtrados", len(df_view))
c2.metric("Yield Médio", f"{df_view[df_view['DY']>0]['DY'].mean():.2f}%")
c3.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
if usar_yahoo and (altas+baixas > 0):
    c4.metric("Tendência (Top 50)", f"{altas} ⬆ vs {baixas} ⬇")
else:
    c4.metric("Tendência", "OFF")

# Setor mais descontado (Menor P/L médio)
try:
    setor_barato = df_view[df_view['PL']>0].groupby('Setor')['PL'].mean().idxmin()
    pl_barato = df_view[df_view['PL']>0].groupby('Setor')['PL'].mean().min()
    c5.metric(f"Setor Descontado", f"{setor_barato} ({pl_barato:.1f}x)")
except: c5.metric("Setor Descontado", "-")

st.divider()

# C. Área Principal
col_list, col_dash = st.columns([1.5, 2.5])

# Seleção
sel_row = None
cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "Momentum": st.column_config.NumberColumn("Mom. (6m)", format="%.1f%%"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%")
}

with col_list:
    st.subheader("📋 Mercado")
    tabs = st.tabs(["Geral", "Dividendos", "Valor", "Magic"])
    
    with tabs[0]: # Geral
        df_show = df_view.sort_values('Liquidez', ascending=False).head(100)
        ev = st.dataframe(df_show[['Ticker','Setor','Preco','Momentum']], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if len(ev.selection.rows)>0: sel_row = df_show.iloc[ev.selection.rows[0]]
        
    with tabs[1]: # Dividendos
        df_show = df_view.nlargest(100, 'DY')
        ev = st.dataframe(df_show[['Ticker','Preco','DY','Bazin_Fair']], column_config={**cfg, "Bazin_Fair": st.column_config.NumberColumn("Teto Bazin", format="%.2f")}, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if len(ev.selection.rows)>0: sel_row = df_show.iloc[ev.selection.rows[0]]
        
    with tabs[2]: # Valor (Graham)
        df_show = df_view[(df_view['Upside']>0)&(df_view['Upside']<500)].nlargest(100, 'Upside')
        ev = st.dataframe(df_show[['Ticker','Preco','Graham_Fair','Upside']], column_config={**cfg, "Graham_Fair": st.column_config.NumberColumn("Valor Justo", format="%.2f")}, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if len(ev.selection.rows)>0: sel_row = df_show.iloc[ev.selection.rows[0]]

    with tabs[3]: # Magic
        df_show = df_view.nsmallest(100, 'Score_Magic')
        ev = st.dataframe(df_show[['Ticker','Preco','EV_EBIT','ROIC','Score_Magic']], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if len(ev.selection.rows)>0: sel_row = df_show.iloc[ev.selection.rows[0]]

# D. Painel de Detalhes
with col_dash:
    if sel_row is not None:
        t = sel_row['Ticker']
        st.markdown(f"## 🔎 <span style='color:#00e676'>{t}</span> | {sel_row['Setor']}", unsafe_allow_html=True)
        
        # Download Histórico 10 Anos
        if usar_yahoo:
            with st.spinner(f"Analisando {t}..."):
                try:
                    tk = yf.Ticker(t+".SA")
                    hist = tk.history(period="10y")
                    
                    if not hist.empty:
                        # Indicadores Técnicos
                        hist['SMA50'] = hist['Close'].rolling(50).mean()
                        hist['SMA200'] = hist['Close'].rolling(200).mean()
                        
                        # Abas de Gráfico
                        g1, g2, g3 = st.tabs(["📈 Price Action", "📊 Fundamentos", "🧠 Comparativo"])
                        
                        with g1:
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='OHLC'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1), name='MM50'), row=1, col=1)
                            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='cyan', width=1), name='MM200'), row=1, col=1)
                            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color='#2d3748'), row=2, col=1)
                            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, title="Histórico 10 Anos")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with g2:
                            c_f1, c_f2 = st.columns(2)
                            # Margens
                            m_data = pd.DataFrame({'Tipo': ['Bruta', 'EBIT', 'Líquida'], 'Margem': [sel_row.get('MargemEbit',0)*1.3, sel_row.get('MargemEbit',0), sel_row.get('MargemLiquida',0)]})
                            fig_m = px.bar(m_data, x='Tipo', y='Margem', color='Tipo', title="Margens (%)", template="plotly_dark")
                            c_f1.plotly_chart(fig_m, use_container_width=True)
                            
                            # Dívida
                            div = sel_row.get('Div_Patrimonio', 0)
                            fig_g = go.Figure(go.Indicator(
                                mode = "gauge+number", value = div,
                                title = {'text': "Dívida / PL"},
                                gauge = {'axis': {'range': [None, 5]}, 'bar': {'color': "red" if div>3 else "green"}}
                            ))
                            fig_g.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20), template="plotly_dark")
                            c_f2.plotly_chart(fig_g, use_container_width=True)
                            
                        with g3:
                            # Scatter Setor
                            df_s = df_view[df_view['Setor'] == sel_row['Setor']]
                            fig_s = px.scatter(df_s, x='PL', y='ROE', size='Liquidez', color='DY', hover_name='Ticker', title=f"Setor: {sel_row['Setor']}", template="plotly_dark")
                            fig_s.add_annotation(x=sel_row['PL'], y=sel_row['ROE'], text=t, showarrow=True, arrowhead=1, font=dict(color='cyan', size=14))
                            st.plotly_chart(fig_s, use_container_width=True)
                            
                    else: st.warning("Dados históricos indisponíveis.")
                except Exception as e: st.error(f"Erro gráfico: {e}")
        else:
            st.warning("Ative 'Dados Técnicos' na barra lateral para ver os gráficos.")
            
        # Métricas Rápidas
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P/L", f"{sel_row['PL']:.1f}x")
        m2.metric("P/VP", f"{sel_row['PVP']:.1f}x")
        m3.metric("ROE", f"{sel_row['ROE']:.1f}%")
        m4.metric("Dívida/PL", f"{sel_row.get('Div_Patrimonio',0):.2f}")
            
    else:
        # Visão Geral (Sem Seleção)
        st.info("👈 Selecione um ativo na tabela.")
        
        st.subheader("Mapa de Mercado")
        try:
            df_tree = df_view.groupby('Setor')[['Liquidez', 'DY']].mean().reset_index()
            df_tree['Qtd'] = df_view.groupby('Setor')['Ticker'].count().values
            fig_tree = px.treemap(df_tree, path=['Setor'], values='Qtd', color='DY', color_continuous_scale='Viridis', title="Setores por Quantidade (Cor = Yield)", template="plotly_dark")
            st.plotly_chart(fig_tree, use_container_width=True)
        except: pass
