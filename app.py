import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Titanium Pro III", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    [data-testid="stMetricValue"] { font-size: 1.2rem; color: #00f2ea; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #161b22; padding: 5px; border-radius: 5px; }
    .stTabs [data-baseweb="tab"] { height: 35px; background-color: transparent; color: #aaa; border: none; font-size: 13px; }
    .stTabs [aria-selected="true"] { background-color: #262d3d !important; color: #00f2ea !important; border-bottom: 2px solid #00f2ea; }
    div.stDataFrame div[data-testid="stDataFrame"] { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro III: Accounting & Momentum")

# --- 2. FUNÇÕES DE DADOS ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data():
    try:
        # Fundamentus Raw
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Mapeamento Expandido
        mapa = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT', 'Liq.2meses': 'Liquidez',
            'Mrg. Líq.': 'MargemLiquida', 'Dív.Brut/ Patr.': 'Div_Patrimonio',
            'Cresc. Rec.5a': 'Cresc_5a', 'Patrim. Líq': 'Patrimonio', 'Ativo': 'Ativos',
            'Margem EBIT': 'MargemEbit', 'P/Ativo': 'P_Ativo', 'PSR': 'PSR',
            'P/Cap.Giro': 'P_CapGiro', 'Liq. Corr.': 'LiqCorrente'
        }
        
        cols = ['Ticker'] + [c for c in mapa.keys() if c in df.columns]
        df = df[cols].copy().rename(columns=mapa, inplace=True)
        
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(clean_float)
            
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        # Classificação Setorial Simplificada
        def get_setor(t):
            t = t.upper()
            if t.startswith(('ITUB','BBDC','BBAS','SANB','BPAC','B3SA','BBSE')): return 'Financeiro'
            if t.startswith(('VALE','CSNA','GGBR','USIM','SUZB','KLBN')): return 'Materiais Básicos'
            if t.startswith(('PETR','PRIO','UGPA','CSAN','RRRP')): return 'Petróleo & Gás'
            if t.startswith(('MGLU','LREN','ARZZ','PETZ','AMER')): return 'Varejo'
            if t.startswith(('WEGE','EMBR','TUPY','RAPT')): return 'Industrial'
            if t.startswith(('TAEE','TRPL','ELET','CPLE','EQTL','CMIG')): return 'Elétricas/Saneamento'
            if t.startswith(('RADL','RDOR','HAPV','FLRY')): return 'Saúde'
            if t.startswith(('CYRE','EZTC','MRVE','TEND')): return 'Construção'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings Quantitativos
        df['Graham'] = np.where((df['PL']>0)&(df['PVP']>0), np.sqrt(22.5 * (df['Preco']/df['PL']) * (df['Preco']/df['PVP'])), 0)
        df['Upside'] = np.where((df['Graham']>0), ((df['Graham']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin'] = np.where(df['DY']>0, df['Preco']*(df['DY']/6), 0)
        
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
            
        return df
    except: return pd.DataFrame()

# --- 3. MOTOR CONTÁBIL (ANÁLISE VERTICAL/HORIZONTAL) ---
def get_financial_deep_dive(ticker):
    """Baixa DRE e Balanço do Yahoo Finance para análise profunda"""
    try:
        stock = yf.Ticker(ticker + ".SA")
        
        # Tenta pegar anual. Se falhar, retorna None
        income = stock.financials
        balance = stock.balance_sheet
        
        if income.empty or balance.empty:
            return None, None
            
        # Transpor para ter datas nas linhas (Ano 1, Ano 2...)
        income = income.T.sort_index(ascending=True)
        balance = balance.T.sort_index(ascending=True)
        
        # --- ANÁLISE VERTICAL (DRE) ---
        # Base 100 = Total Revenue
        if 'Total Revenue' in income.columns:
            income_av = pd.DataFrame()
            income_av['Receita Líquida'] = income['Total Revenue']
            
            # Principais linhas
            cols_dre = ['Cost Of Revenue', 'Gross Profit', 'Operating Expense', 'Operating Income', 'Net Income']
            mapa_dre = {'Cost Of Revenue': 'Custos', 'Gross Profit': 'Lucro Bruto', 
                        'Operating Expense': 'Desp. Operacionais', 'Operating Income': 'EBIT', 'Net Income': 'Lucro Líquido'}
            
            for col in cols_dre:
                if col in income.columns:
                    # Cálculo AV%
                    income_av[f'{mapa_dre[col]} (R$)'] = income[col]
                    income_av[f'{mapa_dre[col]} AV%'] = (income[col] / income['Total Revenue']) * 100
            
            # Formatação para exibição
            income_av = income_av.iloc[-4:] # Últimos 4 anos

        else: income_av = None

        # --- ANÁLISE HORIZONTAL (CRESCIMENTO) ---
        if 'Total Revenue' in income.columns and 'Net Income' in income.columns:
            ah_data = income[['Total Revenue', 'Net Income']].pct_change() * 100
            ah_data.columns = ['Cresc. Receita (%)', 'Cresc. Lucro (%)']
            ah_data = ah_data.iloc[-4:]
        else: ah_data = None
            
        return income_av, ah_data

    except Exception as e:
        return None, None

# --- 4. INTERFACE ---
st.sidebar.header("🎛️ Filtros & Momentum")
usar_tech = st.sidebar.checkbox("📡 Ativar Momentum (Yahoo)", value=True)

with st.spinner('Inicializando Base de Dados...'):
    df = get_market_data()

if df.empty:
    st.error("Erro na conexão B3.")
    st.stop()

# Filtros
busca = st.sidebar.text_input("Ticker", placeholder="PETR4").upper()
setor_f = st.sidebar.selectbox("Setor", ["Todos"] + sorted(df['Setor'].unique().tolist()))
liq_f = st.sidebar.select_slider("Liquidez Mínima", options=[0, 100000, 200000, 1000000, 5000000], value=200000)

df_view = df[df['Liquidez'] >= liq_f].copy()
if setor_f != "Todos": df_view = df_view[df_view['Setor'] == setor_f]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Cálculo Momentum (Só tabela)
if usar_tech:
    with st.spinner("Calculando Momentum (Top 60)..."):
        top_m = df_view.nlargest(60, 'Liquidez')['Ticker'].tolist()
        ts = [t+".SA" for t in top_m]
        try:
            h = yf.download(ts, period="6mo", progress=False)['Adj Close']
            if isinstance(h, pd.Series): h = h.to_frame(name=ts[0])
            res = {}
            for c in h.columns:
                s = h[c].dropna()
                if len(s)>10:
                    ret = ((s.iloc[-1]-s.iloc[0])/s.iloc[0])*100
                    res[c.replace('.SA','')] = ret
            df_view['Momentum'] = df_view['Ticker'].map(res).fillna(0)
        except: df_view['Momentum'] = 0
else: df_view['Momentum'] = 0

# --- LAYOUT DIVIDIDO ---
col_L, col_R = st.columns([1.5, 2.5])

# --- COLUNA ESQUERDA (TABELAS) ---
with col_L:
    st.subheader("📋 Screener")
    t1, t2, t3, t4 = st.tabs(["Geral", "Dividendos", "Valor", "Magic"])
    
    cfg = {
        "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
        "DY": st.column_config.ProgressColumn("DY", format="%.1f%%", min_value=0, max_value=15),
        "Momentum": st.column_config.NumberColumn("Mom.", format="%.1f%%"),
        "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%")
    }
    
    sel = None
    
    def render_table(d, c, k):
        ev = st.dataframe(d, column_config=c, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500, key=k)
        if len(ev.selection.rows)>0: return d.iloc[ev.selection.rows[0]]
        return None

    with t1:
        d = df_view.sort_values('Liquidez', ascending=False).head(100)
        s = render_table(d[['Ticker','Preco','Momentum','DY']], cfg, 't1')
        if s is not None: sel = s
    with t2:
        d = df_view.nlargest(100, 'DY')
        s = render_table(d[['Ticker','Preco','DY','Bazin']], {**cfg, "Bazin": st.column_config.NumberColumn("Teto", format="%.2f")}, 't2')
        if s is not None: sel = s
    with t3:
        d = df_view[(df_view['Upside']>0)&(df_view['Upside']<500)].nlargest(100, 'Upside')
        s = render_table(d[['Ticker','Preco','Graham','Upside']], {**cfg, "Graham": st.column_config.NumberColumn("Justo", format="%.2f")}, 't3')
        if s is not None: sel = s
    with t4:
        d = df_view.nsmallest(100, 'Score_Magic')
        s = render_table(d[['Ticker','Preco','EV_EBIT','ROIC','Score_Magic']], cfg, 't4')
        if s is not None: sel = s

# --- COLUNA DIREITA (DATA CENTER) ---
with col_R:
    if sel is not None:
        tk = sel['Ticker']
        st.markdown(f"## 📊 Análise Profunda: <span style='color:#00f2ea'>{tk}</span>", unsafe_allow_html=True)
        
        # Tabs de Análise
        tab_main, tab_cont, tab_tec = st.tabs(["📈 Resumo & Gráfico", "📑 Análise Contábil (V/H)", "🧠 Indicadores"])
        
        # 1. Resumo & Gráfico
        with tab_main:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço", f"R$ {sel['Preco']:.2f}")
            c2.metric("P/L", f"{sel['PL']:.1f}x")
            c3.metric("ROE", f"{sel['ROE']:.1f}%")
            c4.metric("Dívida/PL", f"{sel.get('Div_Patrimonio',0):.2f}")
            
            if usar_yahoo:
                with st.spinner(f"Baixando histórico {tk}..."):
                    try:
                        h = yf.download(tk+".SA", period="3y", progress=False)
                        if not h.empty:
                            # Tratamento MultiIndex do Yahoo novo
                            if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                            
                            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
                            fig.update_layout(title="Price Action (3 Anos)", template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)
                    except: st.error("Gráfico indisponível.")

        # 2. Análise Contábil (NOVIDADE)
        with tab_cont:
            st.markdown("#### 📉 Análise Vertical (Margens) & Horizontal (Crescimento)")
            if usar_yahoo:
                with st.spinner("Processando Balanços e DRE..."):
                    df_av, df_ah = get_financial_deep_dive(tk)
                    
                    if df_av is not None:
                        st.markdown("**Demonstrativo de Resultados (AV%)**")
                        # Formata para ficar bonito
                        st.dataframe(df_av.style.format("{:,.2f}"), use_container_width=True)
                        
                        # Gráfico de Receita vs Lucro
                        fig_fin = px.bar(df_av, x=df_av.index.year, y=['Lucro Bruto (R$)', 'Lucro Líquido (R$)'], 
                                         barmode='group', title="Evolução de Resultados (R$)", template="plotly_dark")
                        st.plotly_chart(fig_fin, use_container_width=True)
                    else:
                        st.warning("Dados contábeis detalhados não disponíveis para este ativo.")
                        
                    if df_ah is not None:
                        st.markdown("**Crescimento Ano a Ano (AH%)**")
                        st.dataframe(df_ah.style.format("{:,.2f}%").background_gradient(cmap="RdYlGn", vmin=-20, vmax=20), use_container_width=True)
            else:
                st.warning("Ative 'Dados Técnicos' na barra lateral para ver a contabilidade.")

        # 3. Indicadores Extras
        with tab_tec:
            col_i1, col_i2 = st.columns(2)
            # Margens
            m_df = pd.DataFrame({'Margem': ['Bruta', 'EBIT', 'Líquida'], 'Valor': [sel.get('MargemEbit',0)*1.4, sel.get('MargemEbit',0), sel.get('MargemLiquida',0)]})
            fig_m = px.bar(m_data_frame=m_df, x='Margem', y='Valor', color='Margem', title="Estrutura de Margens (%)", template="plotly_dark")
            col_i1.plotly_chart(fig_m, use_container_width=True)
            
            # Liquidez
            liq_data = pd.DataFrame({'Liquidez': ['Corrente', 'Geral'], 'Ratio': [sel.get('LiqCorrente',0), sel.get('LiqCorrente',0)*0.8]})
            fig_l = px.bar(liq_data, x='Liquidez', y='Ratio', title="Índices de Liquidez", template="plotly_dark")
            fig_l.add_hline(y=1, line_dash="dash", line_color="red")
            col_i2.plotly_chart(fig_l, use_container_width=True)

    else:
        st.info("👈 Selecione um ativo na tabela.")
        
        # Dashboard Macro
        st.subheader("Visão Macro")
        try:
            df_tree = df_view.groupby('Setor')[['Liquidez', 'DY']].mean().reset_index()
            df_tree['Qtd'] = df_view.groupby('Setor')['Ticker'].count().values
            fig_t = px.treemap(df_tree, path=['Setor'], values='Qtd', color='DY', color_continuous_scale='Viridis', title="Market Map (Setores)", template="plotly_dark")
            st.plotly_chart(fig_t, use_container_width=True)
        except: pass
