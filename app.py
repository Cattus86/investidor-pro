import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Titanium Pro VI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; color: #00ffbf; font-family: 'Roboto Mono', monospace; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
    div[data-testid="stDataFrame"] div[class*="stDataFrame"] { border: 1px solid #333; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #161b22; padding: 5px; }
    .stTabs [data-baseweb="tab"] { height: 30px; font-size: 12px; color: #ccc; border: none; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro VI: Full Terminal")

# --- 2. FUNÇÕES DE LIMPEZA E DADOS ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try:
            return float(val)
        except:
            return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_full_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Mapeamento
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
        
        # Limpeza
        for col in df.columns:
            if col != 'Ticker':
                if df[col].dtype == object:
                    df[col] = df[col].apply(clean_float)
        
        # Ajustes
        pct_cols = ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']
        for col in pct_cols:
            if col in df.columns:
                if df[col].mean() < 1:
                    df[col] *= 100

        # Setores
        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            if t in ['VIVT','TIMS','LWSA','TOTS']: return 'Tecnologia'
            return 'Geral'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings
        # Correção Graham: Preco Justo = Raiz(22.5 * LPA * VPA)
        # LPA = Preco / PL | VPA = Preco / PVP
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
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
        st.error(f"Erro Dados: {e}")
        return pd.DataFrame()

# --- 3. ANÁLISE CONTÁBIL ---
def get_accounting_analysis(ticker):
    try:
        stock = yf.Ticker(ticker+".SA")
        inc = stock.financials.T.sort_index(ascending=True)
        if inc.empty:
            return None, None
        
        # Análise Vertical
        if 'Total Revenue' in inc.columns:
            av = pd.DataFrame()
            receita = inc['Total Revenue']
            av['Receita'] = receita
            
            campos = {
                'Cost Of Revenue': 'CPV',
                'Gross Profit': 'Lucro Bruto',
                'Operating Income': 'EBIT',
                'Net Income': 'Lucro Líquido'
            }
            
            for en, pt in campos.items():
                if en in inc.columns:
                    av[pt] = inc[en]
                    av[f'{pt} AV%'] = (inc[en] / receita) * 100
            
            # Análise Horizontal
            ah = inc[list(campos.keys()) + ['Total Revenue']].pct_change() * 100
            ah.columns = [f"{c} AH%" for c in ah.columns]
            
            return av.iloc[-4:], ah.iloc[-4:]
            
    except:
        return None, None
    return None, None

# --- 4. MOTOR MOMENTUM ---
@st.cache_data(ttl=1800)
def get_momentum_data(tickers):
    if not tickers:
        return {}
        
    ts = [t+".SA" for t in tickers]
    
    try:
        # Baixa histórico 7 meses
        h = yf.download(ts, period="7mo", progress=False)['Adj Close']
        
        # Tratamento para ticker único (Series -> DataFrame)
        if isinstance(h, pd.Series):
            h = h.to_frame(name=ts[0])
        
        mom_dict = {}
        for c in h.columns:
            s = h[c].dropna()
            if len(s) > 20:
                # Retorno 6 Meses
                r6m = ((s.iloc[-1] - s.iloc[0]) / s.iloc[0]) * 100
                # Proximidade Máxima
                max_p = s.max()
                curr = s.iloc[-1]
                near = (curr / max_p) * 100
                
                score = (r6m + near) / 2
                mom_dict[c.replace('.SA','')] = score
                
        return mom_dict
    except:
        return {}

# --- 5. INTERFACE ---
with st.spinner("Inicializando Terminal..."):
    df_full = get_full_market_data()

if df_full.empty:
    st.error("Sem conexão.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setores = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor = st.selectbox("Setor", setores)
    
    with st.expander("📊 Indicadores", expanded=True):
        liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 2000000, 10000000], value=500000)
        pl_r = st.slider("P/L", -5.0, 50.0, (-5.0, 30.0))
        dy_r = st.slider("DY %", 0.0, 30.0, (0.0, 30.0))
        roe_m = st.slider("ROE Min", -20.0, 50.0, 0.0)
    
    usar_mom = st.checkbox("Calcular Momentum", value=True)

# Filtros
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_r[0], pl_r[1])) &
    (df_full['DY'].between(dy_r[0], dy_r[1])) &
    (df_full['ROE'] >= roe_m)
)
df_view = df_full[mask].copy()

if setor != "Todos":
    df_view = df_view[df_view['Setor'] == setor]
if busca:
    df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Momentum
if usar_mom:
    with st.spinner("Calculando Momentum (Top 50)..."):
        top = df_view.nlargest(50, 'Liquidez')['Ticker'].tolist()
        mom_scores = get_momentum_data(top)
        df_view['Momentum'] = df_view['Ticker'].map(mom_scores).fillna(0)
else:
    df_view['Momentum'] = 0

# --- LAYOUT PRINCIPAL ---
st.subheader(f"📋 Super Screener ({len(df_view)} ativos)")

t1, t2, t3, t4, t5 = st.tabs(["Geral", "🚀 Momentum", "💰 Dividendos", "💎 Valor", "✨ Magic"])

cols_main = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Momentum', 'Graham_Fair', 'Bazin_Fair']

cfg = {
    "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
    "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
    "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=20),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%"),
    "Div_Patrimonio": st.column_config.NumberColumn("Dívida/PL", format="%.2f"),
    "Momentum": st.column_config.NumberColumn("Mom. Score", format="%.0f"),
    "Graham_Fair": st.column_config.NumberColumn("Justo (Graham)", format="R$ %.2f"),
    "Bazin_Fair": st.column_config.NumberColumn("Teto (Bazin)", format="R$ %.2f"),
    "Cresc_5a": st.column_config.NumberColumn("Cresc. 5a", format="%.1f%%")
}

sel_ticker = None

def show_table(df_i, k):
    ev = st.dataframe(
        df_i[cols_main],
        column_config=cfg,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=400,
        key=k
    )
    if len(ev.selection.rows) > 0:
        return df_i.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t1:
    sel_ticker = show_table(df_view.sort_values('Liquidez', ascending=False), 't1')
with t2:
    sel_ticker = show_table(df_view.sort_values('Momentum', ascending=False), 't2')
with t3:
    sel_ticker = show_table(df_view.sort_values('DY', ascending=False), 't3')
with t4:
    df_view['Upside'] = (df_view['Graham_Fair'] - df_view['Preco']) / df_view['Preco']
    sel_ticker = show_table(df_view.sort_values('Upside', ascending=False), 't4')
with t5:
    sel_ticker = show_table(df_view.nsmallest(len(df_view), 'Score_Magic'), 't5')

# --- DETALHES ---
st.divider()

if sel_ticker:
    # Busca linha no DF Full para garantir dados
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    
    st.markdown(f"## 🔬 Análise: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Preço", f"R$ {row['Preco']:.2f}")
    k2.metric("P/L", f"{row['PL']:.1f}x")
    k3.metric("P/VP", f"{row['PVP']:.2f}x")
    k4.metric("ROE", f"{row['ROE']:.1f}%")
    k5.metric("Dívida/PL", f"{row['Div_Patrimonio']:.2f}")
    k6.metric("Liquidez Corr.", f"{row['LiqCorrente']:.2f}")
    
    tab_g, tab_cont, tab_val = st.tabs(["📈 Gráfico 5 Anos", "📑 Contabilidade (AV/AH)", "💎 Valuation"])
    
    with tab_g:
        if usar_mom:
            with st.spinner("Baixando Histórico..."):
                try:
                    h = yf.download(sel_ticker+".SA", period="5y", progress=False)
                    if not h.empty:
                        if isinstance(h.columns, pd.MultiIndex):
                            h.columns = h.columns.droplevel(1)
                        
                        h['SMA200'] = h['Close'].rolling(200).mean()
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='Preço'))
                        fig.add_trace(go.Scatter(x=h.index, y=h['SMA200'], line=dict(color='orange'), name='MM200'))
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.error("Gráfico indisponível.")
        else:
            st.info("Ative o Yahoo na barra lateral.")

    with tab_cont:
        c_av, c_ah = st.columns(2)
        if usar_mom:
            with st.spinner("Processando DRE..."):
                av, ah = get_accounting_analysis(sel_ticker)
                
                with c_av:
                    st.markdown("#### Análise Vertical (Estrutura)")
                    if av is not None:
                        st.dataframe(av[['Receita', 'Lucro Bruto', 'Lucro Bruto AV%', 'Lucro Líquido AV%']].style.format("{:,.2f}"), use_container_width=True)
                        fig_av = px.bar(av, x=av.index.year, y=['Lucro Bruto AV%', 'Lucro Líquido AV%'], barmode='group', template="plotly_dark")
                        st.plotly_chart(fig_av, use_container_width=True)
                    else:
                        st.warning("Sem dados contábeis.")
                
                with c_ah:
                    st.markdown("#### Análise Horizontal (Crescimento)")
                    if ah is not None:
                        st.dataframe(ah.style.format("{:,.2f}%").background_gradient(cmap="RdYlGn", vmin=-20, vmax=20), use_container_width=True)
                    else:
                        st.warning("Sem dados.")
    
    with tab_val:
        c1, c2 = st.columns(2)
        with c1:
            vals = pd.DataFrame({'Modelo': ['Atual', 'Graham', 'Bazin'], 'Valor': [row['Preco'], row['Graham_Fair'], row['Bazin_Fair']]})
            fig_v = px.bar(vals, x='Modelo', y='Valor', color='Modelo', title="Valuation", template="plotly_dark")
            st.plotly_chart(fig_v, use_container_width=True)
        with c2:
            st.markdown("#### Comparativo Setor")
            df_s = df_view[df_view['Setor'] == row['Setor']]
            fig_s = px.scatter(df_s, x='PL', y='ROE', size='Liquidez', color='DY', hover_name='Ticker', title=f"Setor: {row['Setor']}", template="plotly_dark")
            fig_s.add_annotation(x=row['PL'], y=row['ROE'], text="ESTE", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_s, use_container_width=True)

else:
    st.info("👆 Selecione um ativo na tabela.")
