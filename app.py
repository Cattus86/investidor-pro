import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Titanium Pro VII", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; color: #00ffbf; font-family: 'Roboto Mono', monospace; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
    div[data-testid="stDataFrame"] div[class*="stDataFrame"] { border: 1px solid #333; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #161b22; padding: 5px; }
    .stTabs [data-baseweb="tab"] { height: 35px; font-size: 12px; color: #ccc; border: none; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro VII: Stable Core")

# --- 2. MOTOR DE DADOS (FUNDAMENTUS) ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
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
        
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        for col in df.columns:
            if col != 'Ticker':
                if df[col].dtype == object: df[col] = df[col].apply(clean_float)
        
        pct_cols = ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']
        for col in pct_cols:
            if col in df.columns:
                if df[col].mean() < 1: df[col] *= 100

        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND']: return 'Construção'
            return 'Geral'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0)
        
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999

        return df
    except Exception as e:
        st.error(f"Erro Dados: {e}")
        return pd.DataFrame()

# --- 3. MOTOR CONTÁBIL (YAHOO ROBUSTO) ---
def get_accounting_data(ticker):
    """Tenta baixar dados contábeis com fallback de nomes"""
    try:
        stock = yf.Ticker(ticker+".SA")
        inc = stock.financials.T.sort_index(ascending=True)
        
        if inc.empty: return None, None
        
        # Tenta encontrar a coluna de Receita (Yahoo muda os nomes as vezes)
        rev_col = None
        possible_rev = ['Total Revenue', 'Revenue', 'Gross Revenue']
        for c in possible_rev:
            if c in inc.columns:
                rev_col = c
                break
        
        if rev_col:
            av = pd.DataFrame()
            receita = inc[rev_col]
            av['Receita'] = receita
            
            # Mapeia outras colunas
            targets = {'Gross Profit': 'Lucro Bruto', 'Operating Income': 'EBIT', 'Net Income': 'Lucro Líquido'}
            for en, pt in targets.items():
                if en in inc.columns:
                    av[pt] = inc[en]
                    av[f'{pt} %'] = (inc[en] / receita) * 100
            
            # AH
            ah = inc[[rev_col] + [c for c in targets.keys() if c in inc.columns]].pct_change() * 100
            
            return av.iloc[-4:], ah.iloc[-4:]
            
    except: return None, None
    return None, None

# --- 4. MOMENTUM ---
@st.cache_data(ttl=1800)
def get_momentum(tickers):
    if not tickers: return {}
    ts = [t+".SA" for t in tickers]
    try:
        h = yf.download(ts, period="6mo", progress=False)['Adj Close']
        if isinstance(h, pd.Series): h = h.to_frame(name=ts[0])
        res = {}
        for c in h.columns:
            s = h[c].dropna()
            if len(s)>20:
                ret = ((s.iloc[-1]-s.iloc[0])/s.iloc[0])*100
                res[c.replace('.SA','')] = ret
        return res
    except: return {}

# --- 5. INTERFACE ---
with st.spinner("Conectando..."):
    df_full = get_market_data()

if df_full.empty:
    st.error("Sem conexão.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor_op = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor = st.selectbox("Setor", setor_op)
    
    with st.expander("📊 Indicadores", expanded=True):
        liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 2000000, 10000000], value=500000)
        pl_r = st.slider("P/L", -5.0, 50.0, (-5.0, 30.0))
        dy_r = st.slider("DY %", 0.0, 30.0, (0.0, 30.0))
    
    usar_mom = st.checkbox("Calcular Momentum", value=True)

# Filtros
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_r[0], pl_r[1])) &
    (df_full['DY'].between(dy_r[0], dy_r[1]))
)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Momentum
if usar_mom:
    with st.spinner("Momentum (Top 50)..."):
        top = df_view.nlargest(50, 'Liquidez')['Ticker'].tolist()
        mom_scores = get_momentum(top)
        df_view['Momentum'] = df_view['Ticker'].map(mom_scores).fillna(0)
else:
    df_view['Momentum'] = 0

# Layout Principal
st.subheader(f"📋 Market ({len(df_view)})")

t1, t2, t3, t4, t5 = st.tabs(["Geral", "🚀 Momentum", "💰 Dividendos", "💎 Valor", "✨ Magic"])

cols_main = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Momentum', 'Graham_Fair', 'Upside', 'Score_Magic']

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "Momentum": st.column_config.NumberColumn("Mom.", format="%.1f%%"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%")
}

sel_ticker = None

def show_table(d, k):
    # Garante colunas existentes
    cols_exist = [c for c in cols_main if c in d.columns]
    ev = st.dataframe(d[cols_exist], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=400, key=k)
    if len(ev.selection.rows) > 0: return d.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t1: sel_ticker = show_table(df_view.sort_values('Liquidez', ascending=False), 't1')
with t2: sel_ticker = show_table(df_view.sort_values('Momentum', ascending=False), 't2')
with t3: sel_ticker = show_table(df_view.sort_values('DY', ascending=False), 't3')
with t4: sel_ticker = show_table(df_view.sort_values('Upside', ascending=False), 't4')
with t5: sel_ticker = show_table(df_view.nsmallest(len(df_view), 'Score_Magic'), 't5')

# --- ANÁLISE PROFUNDA ---
st.divider()

if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    st.markdown(f"## 🔬 {sel_ticker}")
    
    # 1. KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Preço", f"R$ {row['Preco']:.2f}")
    k2.metric("P/L", f"{row['PL']:.1f}x")
    k3.metric("P/VP", f"{row['PVP']:.2f}x")
    k4.metric("ROE", f"{row['ROE']:.1f}%")
    k5.metric("Dívida/PL", f"{row.get('Div_Patrimonio',0):.2f}")
    
    # 2. Painel
    tab_g, tab_c, tab_v = st.tabs(["📈 Gráfico", "📑 Contabilidade", "⚖️ Comparativo"])
    
    with tab_g:
        if usar_mom:
            with st.spinner("Baixando Gráfico..."):
                try:
                    h = yf.download(sel_ticker+".SA", period="3y", progress=False)
                    if not h.empty:
                        if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                        fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
                        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.warning("Sem dados históricos.")
                except: st.error("Erro gráfico.")
        else: st.info("Ative Yahoo.")

    with tab_c:
        if usar_mom:
            with st.spinner("Baixando DRE..."):
                av, ah = get_accounting_data(sel_ticker)
                c_av, c_ah = st.columns(2)
                if av is not None:
                    c_av.markdown("**Margens (Análise Vertical)**")
                    c_av.dataframe(av.style.format("{:,.2f}"), use_container_width=True)
                else: c_av.warning("Contabilidade indisponível.")
                
                if ah is not None:
                    c_ah.markdown("**Crescimento (Análise Horizontal)**")
                    c_ah.dataframe(ah.style.format("{:,.2f}%"), use_container_width=True)
        else: st.info("Ative Yahoo.")

    with tab_v:
        c1, c2 = st.columns(2)
        with c1:
            vals = pd.DataFrame({'Modelo': ['Atual', 'Graham', 'Bazin'], 'Valor': [row['Preco'], row['Graham_Fair'], row['Bazin_Fair']]})
            fig_v = px.bar(vals, x='Modelo', y='Valor', color='Modelo', title="Valuation", template="plotly_dark")
            st.plotly_chart(fig_v, use_container_width=True)
        with c2:
            st.markdown("#### Setor")
            df_s = df_view[df_view['Setor'] == row['Setor']]
            fig_s = px.scatter(df_s, x='PL', y='ROE', size='Liquidez', color='DY', hover_name='Ticker', title=f"Setor: {row['Setor']}", template="plotly_dark")
            fig_s.add_annotation(x=row['PL'], y=row['ROE'], text="ESTE", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_s, use_container_width=True)

else:
    st.info("👆 Selecione um ativo.")
