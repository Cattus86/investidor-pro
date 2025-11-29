import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO VISUAL QUANT ---
st.set_page_config(page_title="Titanium XIX | Quant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #050505; }
    
    /* Tipografia de Terminal */
    h1, h2, h3, h4 { font-family: 'Roboto', sans-serif; color: #e0e0e0; }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-family: 'Fira Code', monospace;
        font-size: 1.8rem;
        color: #00ffbf;
        font-weight: 700;
    }
    
    /* Cards de Análise */
    .quant-box {
        background-color: #0f1115;
        border: 1px solid #2d333b;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .score-badge {
        font-size: 2rem; font-weight: bold; padding: 5px 15px; border-radius: 50%; border: 3px solid; display: inline-block;
    }
    
    /* Tabelas */
    div[data-testid="stDataFrame"] { border: 1px solid #333; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium XIX: Quant Master")

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
def get_data_engine():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        df.columns = normalize_cols(df.columns)
        
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'pativo': 'P_Ativo', 'pcapgiro': 'P_CapGiro',
            'pebit': 'P_EBIT', 'evebit': 'EV_EBIT', 'evebitda': 'EV_EBITDA', 
            'mrgebit': 'MargemEbit', 'mrgliq': 'MargemLiquida', 'liqcorr': 'LiqCorrente',
            'roic': 'ROIC', 'roe': 'ROE', 'liq2meses': 'Liquidez',
            'patrimliq': 'Patrimonio', 'divbrutpatr': 'Div_Patrimonio',
            'crescrec5a': 'Cresc_5a'
        }
        
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(clean_float)
                
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        req = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'PSR', 'LiqCorrente']
        for c in req: 
            if c not in df.columns: df[c] = 0.0

        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','BRKM']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Energia'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA','ALPA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO','KEPL']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Utilidade Pública'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL','ODPV']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF','DIRR']: return 'Construção'
            if t in ['ABEV','JBSS','BRFS','MRFG','BEEF','SMTO']: return 'Consumo'
            return 'Geral'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings Quantitativos Avançados
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        
        # Fórmula de Greenblatt (Ranking Composto)
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Rank_EV'] = df_m['EV_EBIT'].rank(ascending=True)
            df_m['Rank_ROIC'] = df_m['ROIC'].rank(ascending=False)
            df_m['Score_Magic'] = df_m['Rank_EV'] + df_m['Rank_ROIC']
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else: df['Score_Magic'] = 99999

        return df
    except: return pd.DataFrame()

# --- 3. CÉREBRO QUANT (PIOTROSKI SIMPLIFICADO & RISK AUDIT) ---
def analise_quantitativa_pro(ticker, row, df_full):
    """
    Motor de análise que simula um Hedge Fund Quantitativo.
    Analisa: Piotroski F-Score (Proxy), Valuation Relativo, Risco de Balanço.
    """
    score = 0
    max_score = 10
    
    # 1. Piotroski F-Score (Proxy com dados disponíveis)
    # Lucratividade
    f_score = 0
    f_txt = []
    
    if row['ROE'] > 0: f_score += 1 # Retorno positivo
    if row['LiqCorrente'] > 1: f_score += 1 # Liquidez
    if row['Div_Patrimonio'] < 1: f_score += 1 # Baixa Alavancagem
    if row['MargemLiquida'] > 5: f_score += 1 # Margem Operacional (Proxy)
    
    # Eficiência (Comparativa Setorial)
    peers = df_full[df_full['Setor'] == row['Setor']]
    med_roe = peers['ROE'].median()
    med_margem = peers['MargemLiquida'].median()
    
    if row['ROE'] > med_roe: f_score += 1
    if row['MargemLiquida'] > med_margem: f_score += 1
    
    f_txt.append(f"F-Score (Eficiência): {f_score}/6 pontos nos critérios fundamentalistas.")
    score += (f_score / 6) * 4 # Peso 40%
    
    # 2. Valuation Relativo (vs Setor)
    med_pl = peers[(peers['PL']>0)&(peers['PL']<50)]['PL'].median()
    val_score = 0
    val_txt = []
    
    if row['PL'] > 0:
        if row['PL'] < med_pl * 0.8:
            val_txt.append(f"🟢 **Desconto:** P/L de {row['PL']:.1f}x é {((med_pl-row['PL'])/med_pl)*100:.0f}% menor que a média do setor ({med_pl:.1f}x).")
            val_score += 3
        elif row['PL'] > med_pl * 1.2:
            val_txt.append(f"🔴 **Prêmio:** Negociada com ágio sobre o setor ({row['PL']:.1f}x vs {med_pl:.1f}x).")
        else:
            val_txt.append("🔵 **Justo:** Valuation em linha com os pares.")
            val_score += 1
    
    if row['Upside'] > 30: val_score += 2
    
    score += val_score # Peso até 50% (Max 5 pts)
    
    # 3. Auditoria de Risco (Red Flags)
    risk_penalties = 0
    risks = []
    
    if row['Div_Patrimonio'] > 3.5:
        risks.append("⚠️ **Alavancagem Crítica:** Dívida 3.5x maior que o patrimônio.")
        risk_penalties += 2
    if row['Liquidez'] < 500000:
        risks.append("⚠️ **Liquidez Baixa:** Risco de execução na venda.")
        risk_penalties += 1
    if row['MargemLiquida'] < 2:
        risks.append("⚠️ **Margem Fina:** Qualquer aumento de custo pode virar prejuízo.")
        risk_penalties += 1
    if row['ROE'] < 0:
        risks.append("⚠️ **Destruição de Valor:** ROE negativo.")
        risk_penalties += 2
        
    score -= risk_penalties
    
    # Normalização e Veredito
    score = max(0, min(10, score))
    rating = "NEUTRO"
    if score >= 7.5: rating = "STRONG BUY (Oportunidade)"
    elif score >= 6.0: rating = "BUY (Acumular)"
    elif score <= 3.0: rating = "SELL (Risco Elevado)"
    
    # Dados para Radar
    radar = {
        'Qualidade (F-Score)': (f_score/6)*100,
        'Valuation': min(100, (15/max(row['PL'], 1))*50 + (50 if row['PL']<med_pl else 0)),
        'Segurança': max(0, 100 - (row.get('Div_Patrimonio', 0)*20)),
        'Crescimento': min(100, (row.get('Cresc_5a', 0)/15)*100),
        'Dividendos': min(100, (row['DY']/10)*100)
    }
    
    return score, rating, f_txt + val_txt, risks, radar

# --- 4. INTERFACE ---
with st.spinner("Inicializando Motor Quantitativo..."):
    df_full = get_data_engine()

if df_full.empty:
    st.error("Erro na conexão.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Filtros Quant")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor = st.selectbox("Setor", ["Todos"] + sorted(df_full['Setor'].unique().tolist()))
    liq_min = st.select_slider("Liquidez", options=[0, 200000, 1000000, 5000000], value=200000)
    
    with st.expander("Filtros Avançados", expanded=True):
        pl_r = st.slider("P/L", -5.0, 40.0, (-5.0, 30.0))
        dy_r = st.slider("DY %", 0.0, 20.0, (0.0, 20.0))
        roe_m = st.slider("ROE Min", -10.0, 30.0, 0.0)

mask = (df_full['Liquidez'] >= liq_min) & (df_full['PL'].between(pl_r[0], pl_r[1])) & \
       (df_full['DY'].between(dy_r[0], dy_r[1])) & (df_full['ROE'] >= roe_m)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Layout Principal
c1, c2, c3 = st.columns([3, 1, 1])
c1.subheader(f"📋 Quant Screener ({len(df_view)})")
c2.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
c3.metric("ROE Médio", f"{df_view['ROE'].mean():.1f}%")

# Tabela
cols = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Graham_Fair', 'Upside', 'Quality_Score']
safe_cols = [c for c in cols if c in df_view.columns]

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Quality_Score": st.column_config.ProgressColumn("Quality", min_value=0, max_value=100)
}

ev = st.dataframe(df_view[safe_cols].sort_values('Liquidez', ascending=False), column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=400)

sel_ticker = None
if len(ev.selection.rows) > 0:
    idx = ev.selection.rows[0]
    ticker_val = df_view.sort_values('Liquidez', ascending=False).iloc[idx]['Ticker']
    sel_ticker = ticker_val

st.divider()

# --- PAINEL QUANT MASTER ---
if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    score, rating, insights, risks, radar = analise_quantitativa_pro(sel_ticker, row, df_full)
    
    st.markdown(f"## 🧬 Análise Quantitativa: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    # 1. Score Card
    col_score, col_radar = st.columns([1, 1])
    
    with col_score:
        color = "#00e676" if score >= 7 else "#ffea00" if score >= 4 else "#ff1744"
        st.markdown(f"""
        <div class="quant-box" style="border-left: 5px solid {color};">
            <h4 style="color:#aaa; margin:0;">VEREDITO ALGORÍTMICO</h4>
            <h1 style="color:{color}; margin:0; font-size: 2.5rem;">{rating}</h1>
            <hr style="border-color: #333;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 3rem; font-weight: bold; color: white;">{score:.1f}</span>
                    <span style="color: #888;">/ 10</span>
                </div>
                <div style="text-align: right; color: #aaa; font-size: 0.9rem;">
                    Baseado em Valuation, Qualidade,<br>Saúde Financeira e Setor.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Insights
        with st.container():
            st.markdown("#### 🧠 Racional da Análise")
            for i in insights: st.markdown(f"✅ {i}")
            if risks:
                st.markdown("---")
                for r in risks: st.markdown(f"🚩 {r}")

    with col_radar:
        # Radar Chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(radar.values()), theta=list(radar.keys()),
            fill='toself', name=sel_ticker,
            line_color='#00ffbf', fillcolor='rgba(0, 255, 191, 0.2)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='#555')),
            template="plotly_dark",
            title="Raio-X 360º (Percentil)",
            margin=dict(t=40, b=20, l=40, r=40),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2. Dados Profundos
    tab1, tab2 = st.tabs(["📑 Matriz Fundamentalista", "📈 Gráfico Técnico"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço Justo (Graham)", f"R$ {row['Graham_Fair']:.2f}", delta=f"{row['Upside']:.0f}% Upside")
        c2.metric("Margem Líquida", f"{row['MargemLiquida']:.1f}%")
        c3.metric("ROE vs Setor", f"{row['ROE']:.1f}%")
        c4.metric("Liquidez Corrente", f"{row.get('LiqCorrente', 0):.2f}")
        
    with tab2:
        try:
            h = yf.download(sel_ticker+".SA", period="5y", progress=False)
            if not h.empty:
                if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                st.line_chart(h['Close'], color="#00ffbf", height=300)
        except: st.error("Gráfico indisponível.")

else:
    st.info("👆 Selecione um ativo na tabela.")
