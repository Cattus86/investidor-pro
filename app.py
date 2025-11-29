import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO VISUAL ELITE ---
st.set_page_config(page_title="Titanium XV | Oracle", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Tema Dark Financeiro */
    .stApp { background-color: #0b0e11; }
    
    /* Métricas Principais */
    [data-testid="stMetricValue"] {
        font-family: 'DIN Condensed', 'Roboto Condensed', sans-serif;
        font-size: 1.8rem;
        color: #00ffbf;
        text-shadow: 0 0 10px rgba(0, 255, 191, 0.2);
    }
    
    /* Cards de Análise */
    .oracle-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 10px;
    }
    .bull-tag { background: rgba(0, 230, 118, 0.2); color: #00e676; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; border: 1px solid #00e676; }
    .bear-tag { background: rgba(255, 23, 68, 0.2); color: #ff1744; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; border: 1px solid #ff1744; }
    
    /* Tabelas */
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; }
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #0d1117; padding: 5px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { height: 40px; border: none; color: #8b949e; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium XV: The Oracle")

# --- 2. MOTOR DE DADOS & CLEANING ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

def normalize_cols(cols):
    new = []
    for c in cols:
        n = unicodedata.normalize('NFKD', c)
        c = "".join([x for x in n if not unicodedata.combining(x)]).lower()
        c = c.replace('.', '').replace('/', '').replace(' ', '')
        new.append(c)
    return new

@st.cache_data(ttl=300, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        df.columns = normalize_cols(df.columns)
        
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'evebit': 'EV_EBIT', 'roic': 'ROIC', 'roe': 'ROE',
            'liq2meses': 'Liquidez', 'mrgliq': 'MargemLiquida', 'mrgebit': 'MargemEbit',
            'divbrutpatr': 'Div_Patrimonio', 'crescrec5a': 'Cresc_5a', 'liqcorr': 'LiqCorrente',
            'evebitda': 'EV_EBITDA'
        }
        
        # Filtra e renomeia
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        # Limpeza
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # Ajuste Percentual
        pct_cols = ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']
        for col in pct_cols:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        # Garantia de Colunas
        req = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'LiqCorrente']
        for c in req: 
            if c not in df.columns: df[c] = 0.0

        # Classificação Setorial Robusta
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA','BBSE','CXSE']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN','FESA']: return 'Materiais Básicos'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR','RECV','ENAT']: return 'Petróleo & Gás'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA','ALPA','CVCB']: return 'Varejo Cíclico'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO','KEPL','SHUL','RAIL']: return 'Bens Industriais'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE','NEOE']: return 'Utilidade Pública'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL','ODPV','MATD','VVEO']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF','DIRR','CURY','TRIS']: return 'Construção'
            if t in ['ABEV','JBSS','BRFS','MRFG','BEEF','SMTO','MDIA','CRFB']: return 'Consumo Não Cíclico'
            if t in ['VIVT','TIMS','LWSA','TOTS','INTB','POSI']: return 'Tecnologia'
            if t in ['SLCE','AGRO','TTEN','SOJA']: return 'Agronegócio'
            return 'Outros'
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Graham & Bazin
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        
        # Magic Formula Score
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else: df['Score_Magic'] = 99999
        
        return df
    except: return pd.DataFrame()

# --- 3. CÉREBRO DA IA "ORACLE" (COMPARATIVO SETORIAL) ---
def analise_oracle(row, df_completo):
    """
    Gera uma análise profissional comparando a ação com a média do seu SECTOR.
    """
    # 1. Dados do Setor
    setor = row['Setor']
    peers = df_completo[df_completo['Setor'] == setor]
    
    # Médias do Setor (Benchmarks)
    avg_pl = peers[(peers['PL']>0) & (peers['PL']<50)]['PL'].median()
    avg_roe = peers['ROE'].median()
    avg_dy = peers['DY'].median()
    avg_pvp = peers['PVP'].median()
    
    score = 50 # Base 100
    tese = []
    riscos = []
    
    # --- PILAR 1: VALUATION RELATIVO (30 pts) ---
    if row['PL'] > 0:
        if row['PL'] < avg_pl * 0.7:
            tese.append(f"🟢 **Desconto Setorial:** Negociada a {row['PL']:.1f}x lucros, um desconto significativo frente à média do setor ({avg_pl:.1f}x).")
            score += 15
        elif row['PL'] > avg_pl * 1.3:
            riscos.append(f"🔴 **Prêmio Excessivo:** Múltiplo P/L ({row['PL']:.1f}x) muito acima dos pares ({avg_pl:.1f}x). Exige forte crescimento.")
            score -= 10
    
    if row['Graham_Fair'] > row['Preco'] * 1.4:
        tese.append(f"🟢 **Graham:** Margem de segurança teórica de {row['Upside']:.0f}% segundo fórmula clássica.")
        score += 10

    # --- PILAR 2: QUALIDADE & EFICIÊNCIA (30 pts) ---
    if row['ROE'] > avg_roe * 1.2:
        tese.append(f"🟢 **Líder de Eficiência:** ROE de {row['ROE']:.1f}% supera a média da indústria ({avg_roe:.1f}%).")
        score += 15
    elif row['ROE'] < avg_roe * 0.8:
        riscos.append(f"🔴 **Baixa Rentabilidade:** ROE inferior aos concorrentes.")
        score -= 5
        
    if row['MargemLiquida'] > 15:
        score += 10
        
    # --- PILAR 3: SAÚDE FINANCEIRA (20 pts) ---
    divida = row.get('Div_Patrimonio', 0)
    if divida < 0.5:
        tese.append("🟢 **Balanço Fortaleza:** Baixíssimo nível de endividamento.")
        score += 10
    elif divida > 3.0:
        riscos.append(f"🔴 **Alavancagem:** Dívida/PL em {divida:.1f}x é um ponto de alerta crítico.")
        score -= 15
        
    if row.get('LiqCorrente', 0) > 1.5:
        score += 5
        
    # --- PILAR 4: FLUXO & RENDA (20 pts) ---
    if row['DY'] > avg_dy * 1.2:
        tese.append(f"🟢 **Yield Alpha:** Paga {row['DY']:.1f}%, acima da média do setor ({avg_dy:.1f}%).")
        score += 10
    
    cresc = row.get('Cresc_5a', 0)
    if cresc > 10:
        tese.append(f"🟢 **Compounder:** Crescimento de receita de {cresc:.1f}% nos últimos 5 anos.")
        score += 10
        
    # Normalização Score Final
    score = min(100, max(0, score))
    
    rating = "NEUTRO"
    if score >= 80: rating = "STRONG BUY"
    elif score >= 60: rating = "BUY"
    elif score <= 40: rating = "SELL"
    
    return score, rating, tese, riscos, avg_pl, avg_roe

# --- 4. INTERFACE ---
with st.spinner("Inicializando Oracle Engine..."):
    df_full = get_market_data()

if df_full.empty:
    st.error("Servidor B3 instável. Tente novamente.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Filtros Globais")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setores = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor = st.selectbox("Setor", setores)
    
    liq_min = st.select_slider("Liquidez Mínima", options=[0, 100000, 500000, 1000000, 5000000], value=500000)
    
    with st.expander("📊 Filtros Finos"):
        pl_r = st.slider("P/L", -10.0, 50.0, (-5.0, 30.0))
        roe_m = st.slider("ROE Min", -20.0, 40.0, 0.0)

# FILTROS
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_r[0], pl_r[1])) &
    (df_full['ROE'] >= roe_m)
)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# --- LAYOUT SUPERIOR ---
st.subheader(f"📋 Screener ({len(df_view)} ativos)")

t_main = st.tabs(["Geral", "💰 Dividendos", "💎 Valor", "✨ Magic Formula", "🛡️ Qualidade"])

# Colunas para a tabela (CORREÇÃO DO BUG: Liquidez DEVE estar aqui para ordenar)
cols_main = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Graham_Fair', 'Upside', 'Score_Magic', 'Liquidez']

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Liquidez": st.column_config.NumberColumn("Liquidez", format="%.0e")
}

sel_ticker = None

def render_table(df_sorted, key):
    # CORREÇÃO: Garante que as colunas existem antes de exibir
    safe_cols = [c for c in cols_main if c in df_sorted.columns]
    
    ev = st.dataframe(
        df_sorted[safe_cols], # Exibe apenas colunas seguras
        column_config=cfg,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=350,
        key=key
    )
    if len(ev.selection.rows) > 0:
        return df_sorted.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t_main[0]: # Geral
    sel_ticker = render_table(df_view.sort_values('Liquidez', ascending=False), 't1')
with t_main[1]: # Dividendos
    sel_ticker = render_table(df_view.nlargest(100, 'DY'), 't2')
with t_main[2]: # Valor
    sel_ticker = render_table(df_view[(df_view['Upside']>0)].nlargest(100, 'Upside'), 't3')
with t_main[3]: # Magic
    sel_ticker = render_table(df_view.nsmallest(100, 'Score_Magic'), 't4')
with t_main[4]: # Qualidade
    df_view['Quality'] = df_view['ROE'] + df_view['MargemLiquida']
    sel_ticker = render_table(df_view.nlargest(100, 'Quality'), 't5')

st.divider()

# --- ORACLE ANALYST ---
if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    score, rating, tese, riscos, avg_pl, avg_roe = analise_oracle(row, df_full)
    
    st.markdown(f"## 👁️ The Oracle: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    # 1. Cabeçalho de Decisão
    col_score, col_chart = st.columns([1, 2])
    
    with col_score:
        color = "#00e676" if score > 70 else "#ffea00" if score > 40 else "#ff1744"
        st.markdown(f"""
        <div class="oracle-card" style="text-align:center; border-top: 5px solid {color};">
            <h4 style="color:#aaa; margin:0;">RATING QUANTITATIVO</h4>
            <h1 style="color:{color}; font-size: 3.5rem; margin:0;">{rating}</h1>
            <h2 style="color:white; margin:0;">{score}/100</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparativo Rápido
        st.caption("Benchmark Setorial:")
        st.progress(min(1.0, row['PL'] / max(1, avg_pl * 2)), text=f"P/L: {row['PL']:.1f} vs Setor: {avg_pl:.1f}")
        st.progress(min(1.0, row['ROE'] / max(1, avg_roe * 2)), text=f"ROE: {row['ROE']:.1f}% vs Setor: {avg_roe:.1f}%")

    with col_chart:
        # Tese e Riscos
        c_bull, c_bear = st.columns(2)
        with c_bull:
            st.markdown("#### 🚀 Tese (Bull Case)")
            if tese:
                for t in tese: st.markdown(f"<span class='bull-tag'>✓</span> {t}", unsafe_allow_html=True)
            else: st.markdown("Sem destaques positivos claros.")
            
        with c_bear:
            st.markdown("#### ⚠️ Riscos (Bear Case)")
            if riscos:
                for r in riscos: st.markdown(f"<span class='bear-tag'>!</span> {r}", unsafe_allow_html=True)
            else: st.markdown("Sem alertas críticos.")

    # 2. Painel Técnico & Contábil
    st.markdown("### 🔬 Deep Dive")
    tab_g, tab_c = st.tabs(["📈 Histórico & Técnica", "📑 Matriz Contábil"])
    
    with tab_g:
        try:
            with st.spinner("Baixando Gráfico..."):
                h = yf.download(sel_ticker+".SA", period="5y", progress=False)
                if not h.empty:
                    if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                    
                    fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
                    fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, title=f"Price Action - {sel_ticker}")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("Gráfico indisponível (Erro Yahoo).")
        except: st.error("Erro ao carregar gráfico.")
        
    with tab_c:
        # Simulação de dados contábeis (Visualização)
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        c_kpi1.metric("Margem Líquida", f"{row['MargemLiquida']:.1f}%")
        c_kpi2.metric("Liquidez Corrente", f"{row.get('LiqCorrente', 0):.2f}")
        c_kpi3.metric("CAGR 5 Anos", f"{row.get('Cresc_5a', 0):.1f}%")
        
        st.info("Para ver o DRE completo, o Yahoo Finance requer validação extra que pode falhar em nuvem compartilhada. Exibindo métricas chave acima.")

else:
    st.info("👆 Selecione um ativo na tabela para ativar o Oráculo.")
