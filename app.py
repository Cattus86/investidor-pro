import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Titanium Pro IV", layout="wide", initial_sidebar_state="expanded")

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

st.title("⚡ Titanium Pro IV: Stealth Edition")

# --- 2. MOTOR DE DADOS BLINDADO (BYPASS) ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data_stealth():
    """
    Função 'Stealth' que finge ser um navegador Chrome para evitar bloqueio do Fundamentus.
    """
    url = 'https://www.fundamentus.com.br/resultado.php'
    # Cabeçalho de Navegador Real (O Segredo)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # 1. Requisição Manual
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        
        # 2. Leitura com Pandas (lendo o HTML bruto)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # 3. Tratamento dos Nomes das Colunas
        df.rename(columns={'Papel': 'Ticker', 'Cotação': 'Preco', 'Div.Yield': 'DY', 'Liq.2meses': 'Liquidez'}, inplace=True)
        
        # Mapeamento para nomes internos (Padronização)
        rename_map = {
            'P/L': 'PL', 'P/VP': 'PVP', 'ROE': 'ROE', 'ROIC': 'ROIC', 
            'EV/EBIT': 'EV_EBIT', 'Mrg. Líq.': 'MargemLiquida', 
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a',
            'Patrim. Líq': 'Patrimonio', 'Ativo': 'Ativos'
        }
        # Renomeia o que encontrar
        df.rename(columns=rename_map, inplace=True)
        
        # 4. Limpeza de Dados (Percentuais vêm como string "10,0%")
        for col in df.columns:
            if col != 'Ticker':
                # Verifica se é string para limpar
                if df[col].dtype == object:
                    df[col] = df[col].apply(clean_float)
        
        # Ajuste de Escala (Fundamentus direto já vem em %, às vezes precisa ajustar)
        # Na leitura direta do HTML, o pandas com decimal=',' costuma resolver, 
        # mas vamos garantir.
        
        # 5. Classificação Setorial (Manual)
        def get_setor(t):
            t = t.upper()
            if t.startswith(('ITUB','BBDC','BBAS','SANB','BPAC','B3SA','BBSE')): return 'Financeiro'
            if t.startswith(('VALE','CSNA','GGBR','USIM','SUZB','KLBN')): return 'Materiais Básicos'
            if t.startswith(('PETR','PRIO','UGPA','CSAN','RRRP')): return 'Petróleo & Gás'
            if t.startswith(('MGLU','LREN','ARZZ','PETZ','AMER')): return 'Varejo'
            if t.startswith(('WEGE','EMBR','TUPY','RAPT')): return 'Industrial'
            if t.startswith(('TAEE','TRPL','ELET','CPLE','EQTL','CMIG')): return 'Utilidade Púb.'
            if t.startswith(('RADL','RDOR','HAPV','FLRY')): return 'Saúde'
            if t.startswith(('CYRE','EZTC','MRVE','TEND')): return 'Construção'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # 6. Rankings Quantitativos
        df['Graham'] = np.where((df['PL']>0)&(df['PVP']>0), np.sqrt(22.5 * (df['Preco']/df['PL']) * (df['Preco']/df['PVP'])), 0)
        df['Upside'] = np.where((df['Graham']>0), ((df['Graham']-df['Preco'])/df['Preco'])*100, -999)
        
        # Bazin (Teto 6%) - Se DY vier como 6.0, divide por 100 antes da conta se precisar
        # O clean_float já transforma "6,0%" em 6.0.
        df['Bazin'] = np.where(df['DY']>0, df['Preco']*((df['DY']/100)/0.06), 0) # Ajuste fórmula bazin correta
        
        # Magic Formula Simples
        if 'EV_EBIT' in df.columns and 'ROIC' in df.columns:
            df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999
            
        return df
    except Exception as e:
        st.error(f"Erro detalhado no bypass: {e}")
        return pd.DataFrame()

# --- 3. MOTOR CONTÁBIL (Yahoo Finance) ---
def get_financial_deep_dive(ticker):
    try:
        stock = yf.Ticker(ticker + ".SA")
        income = stock.financials
        if income.empty: return None, None
        
        income = income.T.sort_index(ascending=True)
        
        if 'Total Revenue' in income.columns:
            df_av = pd.DataFrame()
            rec = income['Total Revenue']
            
            # Mapas de colunas comuns do Yahoo (variam as vezes)
            targets = {
                'Lucro Bruto': 'Gross Profit',
                'EBIT': 'Operating Income',
                'Lucro Líquido': 'Net Income'
            }
            
            df_av['Receita'] = rec
            for label, col_y in targets.items():
                if col_y in income.columns:
                    df_av[label] = income[col_y]
                    df_av[f'{label} %'] = (income[col_y] / rec) * 100
            
            return df_av.iloc[-4:], None
        return None, None
    except: return None, None

# --- 4. INTERFACE ---
st.sidebar.header("🎛️ Centro de Comando")
usar_tech = st.sidebar.checkbox("📡 Ativar Yahoo (Momentum)", value=True)

with st.spinner('Conectando ao servidor B3 (Modo Stealth)...'):
    df = get_market_data_stealth()

if df.empty:
    st.error("Falha crítica: O servidor bloqueou mesmo o acesso Stealth. Tente novamente em 1 minuto.")
    st.stop()

# Filtros
busca = st.sidebar.text_input("Ticker", placeholder="PETR4").upper()
setor_f = st.sidebar.selectbox("Setor", ["Todos"] + sorted(df['Setor'].unique().tolist()))
liq_f = st.sidebar.select_slider("Liquidez Mínima", options=[0, 100000, 200000, 1000000, 5000000], value=200000)

df_view = df[df['Liquidez'] >= liq_f].copy()
if setor_f != "Todos": df_view = df_view[df_view['Setor'] == setor_f]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Momentum (Yahoo)
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

# --- DASHBOARD ---
col_L, col_R = st.columns([1.5, 2.5])

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

with col_L:
    st.subheader("📋 Screener")
    t1, t2, t3, t4 = st.tabs(["Geral", "Dividendos", "Valor", "Magic"])
    
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

with col_R:
    if sel is not None:
        tk = sel['Ticker']
        st.markdown(f"## 📊 Análise: <span style='color:#00f2ea'>{tk}</span>", unsafe_allow_html=True)
        
        tab_main, tab_cont, tab_tec = st.tabs(["📈 Gráfico", "📑 Contabilidade", "🧠 KPIs"])
        
        with tab_main:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço", f"R$ {sel['Preco']:.2f}")
            c2.metric("P/L", f"{sel['PL']:.1f}x")
            c3.metric("ROE", f"{sel['ROE']:.1f}%")
            c4.metric("Dívida/PL", f"{sel.get('Div_Patrimonio',0):.2f}")
            
            if usar_yahoo:
                with st.spinner(f"Baixando histórico..."):
                    try:
                        h = yf.download(tk+".SA", period="3y", progress=False)
                        if not h.empty:
                            if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
                            fig.update_layout(title="Price Action (3 Anos)", template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)
                    except: st.error("Gráfico indisponível.")

        with tab_cont:
            if usar_yahoo:
                with st.spinner("Analisando Balanços..."):
                    df_av, _ = get_financial_deep_dive(tk)
                    if df_av is not None:
                        st.markdown("**Demonstrativo de Resultados (Análise Vertical)**")
                        st.dataframe(df_av.style.format("{:,.2f}"), use_container_width=True)
                        fig_fin = px.bar(df_av, x=df_av.index.year, y=['Lucro Bruto', 'Lucro Líquido'], barmode='group', template="plotly_dark")
                        st.plotly_chart(fig_fin, use_container_width=True)
                    else: st.warning("Dados contábeis não disponíveis.")

        with tab_tec:
            col_i1, col_i2 = st.columns(2)
            m_df = pd.DataFrame({'Margem': ['Líquida'], 'Valor': [sel.get('MargemLiquida',0)]})
            fig_m = px.bar(m_df, x='Margem', y='Valor', color='Margem', title="Margem Líquida (%)", template="plotly_dark")
            col_i1.plotly_chart(fig_m, use_container_width=True)

    else:
        st.info("👈 Selecione um ativo na tabela.")
        st.subheader("Mapa Macro")
        try:
            df_tree = df_view.groupby('Setor')[['Liquidez', 'DY']].mean().reset_index()
            df_tree['Qtd'] = df_view.groupby('Setor')['Ticker'].count().values
            fig_t = px.treemap(df_tree, path=['Setor'], values='Qtd', color='DY', color_continuous_scale='Viridis', title="Setores (Cor=Yield)", template="plotly_dark")
            st.plotly_chart(fig_t, use_container_width=True)
        except: pass
