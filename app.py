import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO DE TERMINAL PROFISSIONAL ---
st.set_page_config(page_title="Titanium Pro VII", layout="wide", initial_sidebar_state="expanded")

# CSS Avançado para Visual "Hedge Fund"
st.markdown("""
<style>
    /* Fundo Geral */
    .stApp { background-color: #0b0e11; }
    
    /* Métricas no Topo */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #00ffbf; /* Neon Green */
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #8b949e; }
    
    /* Tabelas Densas */
    div[data-testid="stDataFrame"] div[class*="stDataFrame"] { border: 1px solid #30363d; }
    
    /* Abas Estilizadas */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #161b22; padding: 8px; border-radius: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 35px; background-color: transparent; color: #8b949e; border: none; font-size: 13px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; border-radius: 4px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
    /* Expander */
    .streamlit-expanderHeader { background-color: #161b22; color: white; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro VII: Hedge Fund Terminal")

# --- 2. MOTOR DE DADOS (FUNDAMENTUS BYPASS) ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Mapeamento Completo (25+ Indicadores)
        rename_map = {
            'Papel': 'Ticker', 'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'PSR': 'PSR',
            'Div.Yield': 'DY', 'P/Ativo': 'P_Ativo', 'P/Cap.Giro': 'P_CapGiro',
            'P/EBIT': 'P_EBIT', 'P/Ativ Circ Liq': 'P_AtivCircLiq',
            'EV/EBIT': 'EV_EBIT', 'EV/EBITDA': 'EV_EBITDA', 'Mrg Ebit': 'MargemEbit',
            'Mrg. Líq.': 'MargemLiquida', 'Liq. Corr.': 'LiqCorrente',
            'ROIC': 'ROIC', 'ROE': 'ROE', 'Liq.2meses': 'Liquidez',
            'Patrim. Líq': 'Patrimonio', 'Dív.Brut/ Patr.': 'Div_Patrimonio',
            'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        # Filtra colunas existentes
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        # Limpeza Numérica
        for col in df.columns:
            if col != 'Ticker':
                if df[col].dtype == object: df[col] = df[col].apply(clean_float)
        
        # Ajuste de Escala Percentual
        pct_cols = ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']
        for col in pct_cols:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # Classificação Setorial (Manual Robusta)
        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA','BBSE','CXSE','IRBR']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','BRKM','CMIN']: return 'Materiais Básicos'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR','RECV','ENAT']: return 'Petróleo & Gás'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA','ALPA','CVCB']: return 'Varejo Cíclico'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO','KEPL','SHUL']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE','NEOE']: return 'Utilidade Pública'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL','ODPV']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF','DIRR','CURY']: return 'Construção'
            if t in ['ABEV','JBSS','BRFS','MRFG','BEEF','SMTO','MDIA','CRFB']: return 'Consumo Não Cíclico'
            if t in ['VIVT','TIMS','LWSA','TOTS','INTB']: return 'Tecnologia'
            if t in ['SLCE','AGRO','TTEN','SOJA']: return 'Agronegócio'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Inicializa colunas vitais para não dar KeyError
        df['Momentum'] = 0.0
        
        # Rankings Quantitativos
        # Graham Seguro
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        
        # Bazin
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0)
        
        # Magic Formula
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados fundamentais: {e}")
        return pd.DataFrame()

# --- 3. ANÁLISE CONTÁBIL (YAHOO) ---
def get_deep_dive(ticker):
    try:
        stock = yf.Ticker(ticker+".SA")
        inc = stock.financials.T.sort_index(ascending=True)
        bal = stock.balance_sheet.T.sort_index(ascending=True)
        
        if inc.empty: return None, None
        
        # Análise Vertical e Horizontal
        av = pd.DataFrame()
        if 'Total Revenue' in inc.columns:
            rev = inc['Total Revenue']
            av['Receita'] = rev
            
            targets = {'Gross Profit': 'Lucro Bruto', 'Operating Income': 'EBIT', 'Net Income': 'Lucro Líquido'}
            for en, pt in targets.items():
                if en in inc.columns:
                    av[pt] = inc[en]
                    av[f'{pt} %'] = (inc[en] / rev) * 100
            
            # Crescimento
            ah = inc[list(targets.keys()) + ['Total Revenue']].pct_change() * 100
            
            return av.iloc[-4:], ah.iloc[-4:]
    except: return None, None
    return None, None

# --- 4. MOMENTUM ENGINE ---
@st.cache_data(ttl=1800)
def calc_momentum(tickers):
    if not tickers: return {}
    ts = [t+".SA" for t in tickers]
    try:
        data = yf.download(ts, period="6mo", progress=False)['Adj Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=ts[0])
        
        res = {}
        for c in data.columns:
            s = data[c].dropna()
            if len(s)>20:
                ret = ((s.iloc[-1]-s.iloc[0])/s.iloc[0])*100
                res[c.replace('.SA','')] = ret
        return res
    except: return {}

# --- 5. INTERFACE ---
with st.spinner("Inicializando Terminal..."):
    df_full = get_market_data()

if df_full.empty:
    st.error("Falha na conexão de dados.")
    st.stop()

# --- SIDEBAR: CONTROLE ---
with st.sidebar:
    st.header("🎛️ Filtros Globais")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor_list = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor = st.selectbox("Setor", setor_list)
    
    with st.expander("📊 Indicadores Fundamentalistas", expanded=True):
        liq_min = st.select_slider("Liquidez Mínima", options=[0, 100000, 500000, 2000000, 10000000], value=500000)
        pl_r = st.slider("P/L", -10.0, 50.0, (-5.0, 30.0))
        dy_r = st.slider("Dividend Yield (%)", 0.0, 30.0, (0.0, 30.0))
        roe_min = st.slider("ROE Mínimo (%)", -20.0, 50.0, 0.0)
    
    usar_mom = st.checkbox("Calcular Momentum (Top 50)", value=True)

# FILTRAGEM
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_r[0], pl_r[1])) &
    (df_full['DY'].between(dy_r[0], dy_r[1])) &
    (df_full['ROE'] >= roe_min)
)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# MOMENTUM LOGIC (Evita Crash)
if usar_mom:
    with st.spinner("Calculando Momentum..."):
        # Pega apenas os top 50 filtrados para não travar
        top_list = df_view.nlargest(50, 'Liquidez')['Ticker'].tolist()
        mom_data = calc_momentum(top_list)
        # Mapeia. Quem não tem dado fica com 0
        df_view['Momentum'] = df_view['Ticker'].map(mom_data).fillna(0)
else:
    df_view['Momentum'] = 0

# --- DASHBOARD DE MERCADO ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ativos Filtrados", len(df_view))
c2.metric("Yield Médio", f"{df_view[df_view['DY']>0]['DY'].mean():.2f}%")
c3.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
# Market Breadth
up = len(df_view[df_view['Momentum'] > 0])
down = len(df_view[df_view['Momentum'] < 0])
c4.metric("Tendência (Top 50)", f"{up} 🟢 / {down} 🔴")
try:
    best_sec = df_view.groupby('Setor')['DY'].mean().idxmax()
    c5.metric("Setor (Yield)", best_sec)
except: c5.metric("Setor", "-")

st.divider()

# --- TABELA PRINCIPAL (SUPER SCREENER) ---
st.subheader(f"📋 Market Screener ({len(df_view)} ativos)")

# Definição das colunas que VÃO aparecer (Segurança contra KeyError)
cols_available = df_view.columns.tolist()
cols_desired = ['Ticker', 'Setor', 'Preco', 'Momentum', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Graham_Fair', 'Bazin_Fair', 'Upside', 'Score_Magic']
# Interseção para garantir que só pedimos o que existe
cols_final = [c for c in cols_desired if c in cols_available]

cfg = {
    "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
    "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
    "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=20),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%"),
    "Div_Patrimonio": st.column_config.NumberColumn("Dívida/PL", format="%.2f"),
    "Momentum": st.column_config.NumberColumn("Mom. 6m", format="%.1f%%"),
    "Graham_Fair": st.column_config.NumberColumn("Justo", format="R$ %.2f"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Score_Magic": st.column_config.NumberColumn("Score", format="%d")
}

# Abas de Rankings
t1, t2, t3, t4, t5 = st.tabs(["Geral (Liquidez)", "🚀 Momentum", "💰 Dividendos", "💎 Valor (Graham)", "✨ Magic Formula"])

sel_ticker = None

def show_table(df_i, key):
    # Passamos cols_final para garantir que não dê erro de coluna inexistente
    ev = st.dataframe(
        df_i[cols_final],
        column_config=cfg,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=450,
        key=key
    )
    if len(ev.selection.rows) > 0:
        return df_i.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t1: sel_ticker = show_table(df_view.sort_values('Liquidez', ascending=False), 't1')
with t2: sel_ticker = show_table(df_view.sort_values('Momentum', ascending=False), 't2')
with t3: sel_ticker = show_table(df_view.sort_values('DY', ascending=False), 't3')
with t4: sel_ticker = show_table(df_view.sort_values('Upside', ascending=False), 't4')
with t5: sel_ticker = show_table(df_view.nsmallest(len(df_view), 'Score_Magic'), 't5')

# --- PAINEL DE DETALHES ---
st.divider()

if sel_ticker:
    # Busca a linha completa no DF Original para garantir todos os dados
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    
    st.markdown(f"## 🔬 Raio-X: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    # Métricas
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Preço", f"R$ {row['Preco']:.2f}")
    k2.metric("P/L", f"{row['PL']:.1f}x")
    k3.metric("ROE", f"{row['ROE']:.1f}%")
    k4.metric("Dívida/PL", f"{row.get('Div_Patrimonio',0):.2f}")
    # Momentum com segurança
    mom_val = df_view[df_view['Ticker'] == sel_ticker]['Momentum'].values[0] if 'Momentum' in df_view.columns else 0
    k5.metric("Momentum", f"{mom_val:.1f}%")
    
    tab_g, tab_c, tab_v = st.tabs(["📈 Gráfico 5 Anos", "📑 Contabilidade (AV/AH)", "⚖️ Valuation"])
    
    with tab_g:
        if usar_mom:
            with st.spinner("Baixando Histórico..."):
                try:
                    h = yf.download(sel_ticker+".SA", period="5y", progress=False)
                    if not h.empty:
                        if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                        h['SMA200'] = h['Close'].rolling(200).mean()
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='Preço'))
                        fig.add_trace(go.Scatter(x=h.index, y=h['SMA200'], line=dict(color='orange'), name='MM200'))
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                except: st.error("Gráfico indisponível.")
        else: st.info("Ative Yahoo na sidebar.")

    with tab_c:
        if usar_mom:
            with st.spinner("Analisando DRE..."):
                av, ah = get_deep_dive(sel_ticker)
                c_av, c_ah = st.columns(2)
                if av is not None:
                    c_av.markdown("**Margens (Análise Vertical)**")
                    c_av.dataframe(av.style.format("{:,.2f}"), use_container_width=True)
                    fig_av = px.bar(av, x=av.index.year, y=['Lucro Bruto %', 'Lucro Líquido %'], barmode='group', template="plotly_dark")
                    c_av.plotly_chart(fig_av, use_container_width=True)
                else: c_av.warning("Sem dados contábeis.")
                
                if ah is not None:
                    c_ah.markdown("**Crescimento (Análise Horizontal)**")
                    c_ah.dataframe(ah.style.format("{:,.2f}%").background_gradient(cmap="RdYlGn", vmin=-20, vmax=20), use_container_width=True)

    with tab_v:
        c1, c2 = st.columns(2)
        vals = pd.DataFrame({'Modelo': ['Atual', 'Graham', 'Bazin'], 'Valor': [row['Preco'], row['Graham_Fair'], row['Bazin_Fair']]})
        fig_v = px.bar(vals, x='Modelo', y='Valor', color='Modelo', title="Valuation Comparativo", template="plotly_dark")
        c1.plotly_chart(fig_v, use_container_width=True)
        
        # Scatter do Setor
        df_setor = df_view[df_view['Setor'] == row['Setor']]
        fig_s = px.scatter(df_setor, x='PL', y='ROE', size='Liquidez', color='DY', hover_name='Ticker', title=f"Setor: {row['Setor']}", template="plotly_dark")
        fig_s.add_annotation(x=row['PL'], y=row['ROE'], text="ESTE", showarrow=True, arrowhead=1)
        c2.plotly_chart(fig_s, use_container_width=True)

else:
    st.info("👆 Selecione uma ação na tabela para ver a Análise Profunda.")
