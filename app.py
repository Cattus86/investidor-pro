import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO DE TERMINAL (SOVEREIGN UI) ---
st.set_page_config(page_title="Titanium XVIII | Sovereign", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Fundo Dark Profissional */
    .stApp { background-color: #050505; }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.6rem;
        color: #00ffea; /* Cyan Neon */
        font-weight: 700;
        text-shadow: 0 0 15px rgba(0, 255, 234, 0.2);
    }
    
    /* Card de Relatório */
    .report-card {
        background-color: #111;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .report-header {
        font-size: 1.1rem; font-weight: bold; color: #fff;
        border-bottom: 2px solid #00ffea; padding-bottom: 10px; margin-bottom: 15px;
    }
    .bull-text { color: #00e676; font-weight: 500; }
    .bear-text { color: #ff1744; font-weight: 500; }
    .neutral-text { color: #888; }
    
    /* Tabelas */
    div[data-testid="stDataFrame"] { border: 1px solid #333; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium XVIII: Sovereign Analyst")

# --- 2. MOTOR DE DADOS ---
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

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        df.columns = normalize_cols(df.columns)
        
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'pativo': 'P_Ativo', 'pcapgiro': 'P_CapGiro',
            'pebit': 'P_EBIT', 'evebit': 'EV_EBIT', 'roic': 'ROIC', 'roe': 'ROE',
            'liq2meses': 'Liquidez', 'mrgliq': 'MargemLiquida', 'mrgebit': 'MargemEbit',
            'divbrutpatr': 'Div_Patrimonio', 'crescrec5a': 'Cresc_5a', 'liqcorr': 'LiqCorrente'
        }
        
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        req_cols = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'PSR', 'LiqCorrente']
        for c in req_cols: 
            if c not in df.columns: df[c] = 0.0

        # Classificação Setorial Aprimorada
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA','BBSE']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR','RECV']: return 'Petróleo & Gás'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA','ALPA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO','KEPL','SHUL']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Utilidade Pública'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL','ODPV','MATD']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF','DIRR','CURY']: return 'Construção'
            if t in ['ABEV','JBSS','BRFS','MRFG','BEEF','SMTO','MDIA']: return 'Consumo'
            if t in ['VIVT','TIMS','LWSA','TOTS','INTB','POSI']: return 'Tecnologia'
            if t in ['SLCE','AGRO','TTEN','SOJA']: return 'Agronegócio'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings Seguros
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        
        # Quality Score
        df['Quality_Score'] = (df['ROE'].rank(pct=True)*0.4 + df['MargemLiquida'].rank(pct=True)*0.3 + (df['Div_Patrimonio']*-1).rank(pct=True)*0.3) * 100

        return df
    except: return pd.DataFrame()

# --- 3. CÉREBRO SOBERANO (IA DE TESE COM CONTEXTO SETORIAL) ---
def sovereign_analysis(ticker, row, df_full):
    """
    Gera tese comparando com pares do setor.
    CORREÇÃO: Lógica de strings segura.
    """
    # 1. Benchmarks do Setor
    peers = df_full[df_full['Setor'] == row['Setor']]
    med_pl = peers[(peers['PL']>0)&(peers['PL']<50)]['PL'].median()
    med_roe = peers['ROE'].median()
    med_dy = peers['DY'].median()
    
    score = 50 # Base 100
    report = {"Positivos": [], "Negativos": [], "Neutros": []}
    
    # --- ANÁLISE DE VALUATION (Relativo e Absoluto) ---
    # PL Absoluto
    if row['PL'] <= 0:
        report["Negativos"].append(f"Prejuízo: A empresa não reportou lucro nos últimos 12 meses (P/L Negativo).")
        score -= 20
    elif row['PL'] < 5:
        report["Positivos"].append(f"Deep Value: P/L de {row['PL']:.1f}x sugere extremo desconto absoluto.")
        score += 15
    
    # PL Relativo (vs Setor)
    if row['PL'] > 0:
        if row['PL'] < med_pl * 0.8:
            report["Positivos"].append(f"Desconto Setorial: Negociada abaixo da média do setor de {row['Setor']} ({med_pl:.1f}x).")
            score += 10
        elif row['PL'] > med_pl * 1.3:
            report["Negativos"].append(f"Prêmio Setorial: Mais cara que seus pares ({med_pl:.1f}x). O mercado exige alto crescimento.")
            score -= 5

    # --- ANÁLISE DE QUALIDADE (ROE) ---
    if row['ROE'] > med_roe * 1.2:
        report["Positivos"].append(f"Líder de Eficiência: ROE de {row['ROE']:.1f}% supera a média do setor ({med_roe:.1f}%).")
        score += 15
    elif row['ROE'] < med_roe * 0.8:
        report["Negativos"].append(f"Eficiência Baixa: ROE inferior aos concorrentes ({med_roe:.1f}%).")
        score -= 10
        
    if row['MargemLiquida'] > 15:
        report["Positivos"].append("Margens Robustas: Margem Líquida acima de 15%, indicando poder de preço.")
        score += 5

    # --- RISCO & DÍVIDA ---
    divida = row.get('Div_Patrimonio', 0)
    if divida > 3.0:
        report["Negativos"].append(f"Alavancagem Crítica: Dívida/PL de {divida:.2f}x é arriscada.")
        score -= 15
    elif divida < 0.5:
        report["Positivos"].append("Balanço Sólido: Baixíssimo endividamento, preparada para crises.")
        score += 10

    # --- DIVIDENDOS ---
    if row['DY'] > med_dy * 1.2 and row['DY'] > 6:
        report["Positivos"].append(f"Top Yield: Paga {row['DY']:.1f}%, acima da média do setor ({med_dy:.1f}%).")
        score += 10
        
    # Classificação Final
    score = max(0, min(100, score))
    rating = "NEUTRO"
    if score >= 75: rating = "COMPRA FORTE 🚀"
    elif score >= 60: rating = "COMPRA ✅"
    elif score <= 30: rating = "VENDA ❌"
    
    # Dados para Radar (0 a 100)
    radar_data = {
        'Valuation': min(100, (15/max(row['PL'],1))*50 + (1 if row['PL']<med_pl else 0)*50),
        'Qualidade': min(100, (row['ROE']/20)*100),
        'Dividendos': min(100, (row['DY']/12)*100),
        'Segurança': min(100, (1/max(divida,0.1))*50),
        'Crescimento': min(100, (row.get('Cresc_5a',0)/20)*100),
        'Momentum': 50 # Placeholder (poderia vir do Yahoo)
    }
    
    return score, rating, report, radar_data

# --- 4. INTERFACE ---
with st.spinner("Inicializando Sovereign Terminal..."):
    df_full = get_market_data()

if df_full.empty:
    st.error("Erro de conexão.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor_op = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor = st.selectbox("Setor", setor_op)
    
    liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 1000000, 5000000], value=500000)
    
    st.info("ℹ️ Dados fundamentalistas com delay de 15min.")

# Filtros
mask = (df_full['Liquidez'] >= liq_min)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Layout Principal
c1, c2, c3 = st.columns([3, 1, 1])
c1.subheader(f"📋 Market Screener ({len(df_view)} ativos)")
c2.metric("P/L Setor", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
c3.metric("ROE Setor", f"{df_view['ROE'].mean():.1f}%")

t_main = st.tabs(["Geral", "💰 Dividendos", "💎 Valor", "✨ Magic Formula", "🛡️ Qualidade", "🚀 Crescimento"])

# Config Tabela
cols_main = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Graham_Fair', 'Upside', 'Score_Magic', 'Quality_Score']
safe_cols = [c for c in cols_main if c in df_view.columns]

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Quality_Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100)
}

sel_ticker = None
def render_tab(df_in, key):
    ev = st.dataframe(df_in[safe_cols], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=350, key=key)
    if len(ev.selection.rows) > 0: return df_in.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t_main[0]: sel_ticker = render_tab(df_view.sort_values('Liquidez', ascending=False), 't1')
with t_main[1]: sel_ticker = render_tab(df_view.nlargest(100, 'DY'), 't2')
with t_main[2]: sel_ticker = render_tab(df_view[(df_view['Upside']>0)].nlargest(100, 'Upside'), 't3')
with t_main[3]: sel_ticker = render_table(df_view.nsmallest(100, 'Score_Magic'), 't4') if 'Score_Magic' in df_view.columns else st.write("Magic Indisponível")
with t_main[4]: sel_ticker = render_tab(df_view.nlargest(100, 'Quality_Score'), 't5')
with t_main[5]: sel_ticker = render_tab(df_view.nlargest(100, 'Cresc_5a'), 't6')

st.divider()

# --- SOVEREIGN ANALYST PANEL ---
if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    score, rating, report, radar_vals = sovereign_analysis(sel_ticker, row, df_full)
    
    st.markdown(f"## 🤖 Sovereign AI: <span style='color:#00ffea'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    col_dash, col_radar = st.columns([1.5, 1])
    
    with col_dash:
        # Score Banner
        color = "#00e676" if score >= 70 else "#ffea00" if score >= 40 else "#ff1744"
        st.markdown(f"""
        <div style="background-color: #111; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin:0; color: white;">{rating}</h2>
                <span style="color: #888;">Recomendação Quantitativa</span>
            </div>
            <div style="text-align: right;">
                <h1 style="margin:0; color: {color}; font-size: 3rem;">{score}</h1>
                <span style="color: #888;">Score / 100</span>
            </div>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        # Relatório de Texto
        with st.container():
            st.markdown("#### 📝 Tese de Investimento (Comparativa)")
            if report['Positivos']:
                for p in report['Positivos']: st.markdown(f"✅ {p}")
            else: st.markdown("⚠️ Sem pontos fortes óbvios.")
            
            if report['Negativos']:
                st.markdown("---")
                for n in report['Negativos']: st.markdown(f"❌ {n}")
                
    with col_radar:
        # Radar Chart
        categories = list(radar_vals.keys())
        values = list(radar_vals.values())
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself', name=sel_ticker,
            line_color='#00ffea', fillcolor='rgba(0, 255, 234, 0.2)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='#888')),
            template="plotly_dark",
            title=dict(text="Raio-X 360º", x=0.5),
            margin=dict(t=40, b=20, l=40, r=40),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Gráfico de Preço (Opcional para não travar)
    with st.expander("📈 Ver Histórico de Preços (Yahoo Finance)"):
        try:
            with st.spinner("Baixando..."):
                h = yf.download(sel_ticker+".SA", period="2y", progress=False)
                if not h.empty:
                    if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                    st.line_chart(h['Close'], color="#00ffea")
                else: st.warning("Sem dados históricos.")
        except: st.error("Erro no gráfico.")

else:
    st.info("👆 Clique em uma ação na tabela para ativar o Sovereign Analyst.")
