import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO DE TERMINAL (VISUAL BLOOMBERG) ---
st.set_page_config(page_title="Titanium Pro VIII", layout="wide", initial_sidebar_state="expanded")

# CSS Avançado
st.markdown("""
<style>
    /* Fundo Dark Profissional */
    .stApp { background-color: #0e1117; }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.5rem;
        color: #00e676; /* Verde Terminal */
        font-weight: 700;
    }
    
    /* Tabelas Compactas */
    .stDataFrame { border: 1px solid #30363d; border-radius: 5px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: #161b22; padding: 5px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { height: 35px; border: none; color: #8b949e; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    
    /* Títulos */
    h1, h2, h3 { color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro VIII: Market Terminal")

# --- 2. MOTOR DE DADOS "CAMALEÃO" (Evita Bloqueio) ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_data_engine():
    # URL Oficial
    url = 'https://www.fundamentus.com.br/resultado.php'
    
    # Headers para fingir ser um navegador real (Bypass de segurança)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    
    try:
        # Requisição direta
        r = requests.get(url, headers=headers)
        r.raise_for_status() # Garante que não foi erro 403/404
        
        # Leitura
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Mapeamento Seguro
        rename_map = {
            'Papel': 'Ticker', 'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 
            'Div.Yield': 'DY', 'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        # Filtra apenas o que veio
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        # Limpeza Numérica Robusta
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
        
        # Ajuste de Escala
        pct_cols = ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'Cresc_5a']
        for col in pct_cols:
            if col in df.columns and df[col].mean() < 1:
                df[col] *= 100

        # Classificação Setorial (Manual)
        def classify_sector(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA','BBSE']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            return 'Geral'
        
        df['Setor'] = df['Ticker'].apply(classify_sector)
        
        # Rankings Quant
        # Graham Seguro
        df['Graham_Fair'] = 0.0
        mask_graham = (df['PL'] > 0) & (df['PVP'] > 0)
        df.loc[mask_graham, 'Graham_Fair'] = np.sqrt(22.5 * (df.loc[mask_graham, 'Preco']/df.loc[mask_graham, 'PL']) * (df.loc[mask_graham, 'Preco']/df.loc[mask_graham, 'PVP']))
        
        df['Upside'] = np.where((df['Graham_Fair']>0) & (df['Preco']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0)
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao conectar com a B3: {e}")
        return pd.DataFrame()

# --- 3. MOTOR CONTÁBIL E GRÁFICO (YAHOO) ---
def get_details(ticker):
    try:
        t_sa = ticker + ".SA"
        stock = yf.Ticker(t_sa)
        
        # 1. Gráfico
        hist = stock.history(period="5y")
        
        # 2. Contábil
        inc = stock.financials.T.sort_index(ascending=True)
        if not inc.empty and 'Total Revenue' in inc.columns:
            inc['Margem Liquida %'] = (inc['Net Income'] / inc['Total Revenue']) * 100
            inc = inc.iloc[-4:] # Últimos 4 anos
        else:
            inc = None
            
        return hist, inc
    except:
        return None, None

# --- 4. INTERFACE ---
with st.spinner("Inicializando Terminal Quantitativo..."):
    df_full = get_data_engine()

if df_full.empty:
    st.warning("Servidor de dados instável. Tente recarregar a página (F5).")
    st.stop()

# --- SIDEBAR: FILTROS ---
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor = st.selectbox("Setor", ["Todos"] + sorted(df_full['Setor'].unique().tolist()))
    
    with st.expander("📊 Indicadores", expanded=True):
        liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 1000000, 10000000], value=500000)
        pl_range = st.slider("P/L", -10.0, 50.0, (-5.0, 30.0))
        dy_range = st.slider("DY %", 0.0, 30.0, (0.0, 30.0))
    
    usar_graficos = st.checkbox("Carregar Gráficos e Detalhes", value=True)

# FILTRAGEM
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_range[0], pl_range[1])) &
    (df_full['DY'].between(dy_range[0], dy_range[1]))
)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# --- DASHBOARD ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ativos Filtrados", len(df_view))
c2.metric("Yield Médio", f"{df_view[df_view['DY']>0]['DY'].mean():.2f}%")
c3.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
try:
    best = df_view.groupby('Setor')['DY'].mean().idxmax()
    c4.metric("Setor (Yield)", best)
except: c4.metric("Setor", "-")

st.divider()

# --- LAYOUT PRINCIPAL ---
col_L, col_R = st.columns([1.5, 2.5])

# CONFIGURAÇÃO DA TABELA
cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=20),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Graham_Fair": st.column_config.NumberColumn("Justo", format="R$ %.2f")
}

# Variável de Seleção
sel_ticker = None

with col_L:
    st.subheader("📋 Market Screener")
    t1, t2, t3 = st.tabs(["Geral", "💰 Dividendos", "💎 Valor"])
    
    # Função segura de renderização
    def render_tab(d, k):
        # Garante que as colunas existem
        cols_safe = [c for c in ['Ticker', 'Preco', 'DY', 'PL', 'PVP', 'Graham_Fair', 'Upside'] if c in d.columns]
        ev = st.dataframe(
            d[cols_safe], 
            column_config=cfg, 
            use_container_width=True, 
            hide_index=True, 
            on_select="rerun", 
            selection_mode="single-row",
            height=500,
            key=k
        )
        if len(ev.selection.rows) > 0:
            return d.iloc[ev.selection.rows[0]]['Ticker']
        return None

    with t1: sel_ticker = render_tab(df_view.sort_values('Liquidez', ascending=False), 't1')
    with t2: sel_ticker = render_tab(df_view.nlargest(100, 'DY'), 't2')
    with t3: sel_ticker = render_tab(df_view.sort_values('Upside', ascending=False), 't3')

# --- PAINEL DE DETALHES ---
with col_R:
    if sel_ticker:
        # Busca linha segura
        row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
        
        st.markdown(f"## 🔬 Análise: <span style='color:#00e676'>{sel_ticker}</span>", unsafe_allow_html=True)
        
        # KPIs Rápidos
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Preço", f"R$ {row['Preco']:.2f}")
        m2.metric("P/L", f"{row['PL']:.1f}x")
        m3.metric("ROE", f"{row['ROE']:.1f}%")
        m4.metric("Dívida/PL", f"{row.get('Div_Patrimonio', 0):.2f}")
        
        if usar_graficos:
            # Gráficos e Dados Profundos
            tg, tc, tv = st.tabs(["📈 Gráfico Pro", "📑 Balanço (DRE)", "⚖️ Valuation"])
            
            with tg:
                with st.spinner("Carregando Gráfico..."):
                    hist, _ = get_details(sel_ticker)
                    if hist is not None and not hist.empty:
                        # Ajuste MultiIndex (problema comum do Yahoo recente)
                        if isinstance(hist.columns, pd.MultiIndex):
                            hist.columns = hist.columns.droplevel(1)
                            
                        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
                        fig.update_layout(title="Histórico 5 Anos", template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Gráfico indisponível no momento.")
            
            with tc:
                with st.spinner("Baixando DRE..."):
                    _, dre = get_details(sel_ticker)
                    if dre is not None:
                        c_av1, c_av2 = st.columns(2)
                        with c_av1:
                            st.markdown("#### Margens")
                            st.dataframe(dre[['Total Revenue', 'Margem Liquida %']].style.format("{:,.2f}"), use_container_width=True)
                        with c_av2:
                            st.markdown("#### Evolução Visual")
                            fig_dre = px.bar(dre, x=dre.index.year, y=['Total Revenue', 'Net Income'], barmode='group', template="plotly_dark", title="Receita vs Lucro")
                            st.plotly_chart(fig_dre, use_container_width=True)
                    else:
                        st.warning("Dados contábeis indisponíveis para este ativo.")
            
            with tv:
                c_v1, c_v2 = st.columns(2)
                vals = pd.DataFrame({'Modelo': ['Atual', 'Graham', 'Bazin'], 'Valor': [row['Preco'], row['Graham_Fair'], row['Bazin_Fair']]})
                fig_v = px.bar(vals, x='Modelo', y='Valor', color='Modelo', title="Valuation", template="plotly_dark")
                c_v1.plotly_chart(fig_v, use_container_width=True)
                
                # Scatter do Setor
                df_setor = df_view[df_view['Setor'] == row['Setor']]
                fig_s = px.scatter(df_setor, x='PL', y='ROE', size='Liquidez', color='DY', hover_name='Ticker', title=f"Setor: {row['Setor']}", template="plotly_dark")
                fig_s.add_annotation(x=row['PL'], y=row['ROE'], text="AQUI", showarrow=True, arrowhead=1)
                c_v2.plotly_chart(fig_s, use_container_width=True)
                
        else:
            st.info("Ative a opção 'Carregar Gráficos' na barra lateral para ver detalhes.")
            
    else:
        st.info("👈 Selecione um ativo na tabela para abrir o Painel de Controle.")
