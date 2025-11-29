import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO (VISUAL ULTRA MODERNO) ---
st.set_page_config(page_title="Titanium Pro XIV", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    
    /* Métricas Neon */
    [data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.4rem;
        color: #00ffbf;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 191, 0.3);
    }
    
    /* Tabelas e Containers */
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
    .st-emotion-cache-1r6slb0 { border: 1px solid #30363d; border-radius: 8px; padding: 15px; background-color: #161b22; }
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: #161b22; padding: 5px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { height: 35px; border: none; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    
    /* Badges do Analista */
    .badge-bull { background-color: #1a3c28; color: #4caf50; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; border: 1px solid #4caf50; }
    .badge-bear { background-color: #3c1a1a; color: #f44336; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; border: 1px solid #f44336; }
    .badge-neutral { background-color: #2d333b; color: #adbac7; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; border: 1px solid #adbac7; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro XIV: AI Architect")

# --- 2. MOTOR DE DADOS ROBUSTO ---
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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        df.columns = normalize_cols(df.columns)
        
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'evebit': 'EV_EBIT', 'roic': 'ROIC', 'roe': 'ROE',
            'liq2meses': 'Liquidez', 'mrgliq': 'MargemLiquida', 'mrgebit': 'MargemEbit',
            'divbrutpatr': 'Div_Patrimonio', 'crescrec5a': 'Cresc_5a', 'liqcorr': 'LiqCorrente'
        }
        
        # Safe Rename
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        # Cleaning
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # Scaling
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        # Fill NaN
        cols_num = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'PSR', 'LiqCorrente']
        for c in cols_num:
            if c not in df.columns: df[c] = 0.0
            
        # Setores
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            return 'Outros'
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings Proprietários
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0)
        
        # Magic Formula Simples
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else: df['Score_Magic'] = 99999
        
        return df
    except: return pd.DataFrame()

# --- 3. CÉREBRO DA IA 2.0 (ANÁLISE DE TESE) ---
def gerar_tese_investimento(row):
    bull_case = []
    bear_case = []
    tags = []
    score = 5 # Neutro
    
    # --- 1. VALUATION ---
    if row['PL'] > 0 and row['PL'] < 6:
        bull_case.append(f"Múltiplo de lucro extremamente atrativo (P/L {row['PL']:.1f}x).")
        tags.append("VALUE")
        score += 2
    elif row['PL'] > 25:
        bear_case.append("Preço exige crescimento agressivo para se justificar (P/L alto).")
        score -= 1
        
    if row['Graham_Fair'] > row['Preco'] * 1.3:
        bull_case.append(f"Desconto de {((row['Graham_Fair']-row['Preco'])/row['Preco'])*100:.0f}% em relação ao Valor de Graham.")
        score += 1
        
    # --- 2. QUALIDADE & EFICIÊNCIA ---
    if row['ROE'] > 20:
        bull_case.append(f"Rentabilidade de elite (ROE {row['ROE']:.1f}%), indicando forte vantagem competitiva.")
        tags.append("HIGH QUALITY")
        score += 2
    elif row['ROE'] < 5:
        bear_case.append("Empresa destrói valor ou tem baixa eficiência (ROE baixo).")
        score -= 2
        
    if row['MargemLiquida'] > 15:
        bull_case.append("Margens líquidas robustas, protegendo contra inflação de custos.")
        
    # --- 3. RISCOS & SAÚDE ---
    divida = row.get('Div_Patrimonio', 0)
    if divida > 3.0:
        bear_case.append(f"Alavancagem perigosa ({divida:.2f}x Patrimônio). Risco em cenário de juros altos.")
        tags.append("HIGH DEBT")
        score -= 3
    elif divida < 0.5:
        bull_case.append("Estrutura de capital sólida e baixo endividamento.")
        tags.append("SAFE")
        score += 1
        
    # --- 4. DIVIDENDOS (COM TRAP CHECK) ---
    if row['DY'] > 12:
        if divida > 2:
            bear_case.append(f"⚠️ **DIVIDEND TRAP:** Yield de {row['DY']:.1f}% com dívida alta é insustentável.")
            score -= 3
        else:
            bull_case.append(f"Yield excepcional de {row['DY']:.1f}% com balanço controlado.")
            tags.append("CASH COW")
            score += 2
    elif row['DY'] > 6:
        bull_case.append(f"Bom gerador de renda passiva ({row['DY']:.1f}%).")
        score += 1
        
    # Score Final (0 a 10)
    score = max(0, min(10, score))
    
    return bull_case, bear_case, tags, score

# --- 4. INTERFACE ---
with st.spinner("Inicializando IA de Mercado..."):
    df_full = get_market_data()

if df_full.empty:
    st.error("Sem conexão.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setores = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor = st.selectbox("Setor", setores)
    
    with st.expander("📊 Filtros Quant", expanded=True):
        liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 1000000, 5000000], value=500000)
        pl_r = st.slider("P/L", -10.0, 50.0, (-5.0, 30.0))
        dy_r = st.slider("DY %", 0.0, 20.0, (0.0, 20.0))
        roe_m = st.slider("ROE Min", -20.0, 40.0, 0.0)

# FILTRAGEM
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_r[0], pl_r[1])) &
    (df_full['DY'].between(dy_r[0], dy_r[1])) &
    (df_full['ROE'] >= roe_m)
)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# --- LAYOUT PRINCIPAL ---
c1, c2, c3 = st.columns([3, 1, 1])
c1.subheader(f"📋 Screener ({len(df_view)} ativos)")
c2.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
c3.metric("ROE Médio", f"{df_view['ROE'].mean():.1f}%")

# TABELA
cols_main = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Graham_Fair', 'Upside', 'Score_Magic']
safe_cols = [c for c in cols_main if c in df_view.columns]

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%"),
    "Div_Patrimonio": st.column_config.NumberColumn("Dívida/PL", format="%.2f"),
    "Graham_Fair": st.column_config.NumberColumn("Justo", format="R$ %.2f"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Cresc_5a": st.column_config.NumberColumn("CAGR 5a", format="%.1f%%")
}

ev = st.dataframe(
    df_view[safe_cols].sort_values('Liquidez', ascending=False),
    column_config=cfg,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    height=350
)

# --- PAINEL DO ANALISTA IA (EMBAIXO DA TABELA) ---
st.divider()

sel_ticker = None
if len(ev.selection.rows) > 0:
    idx = ev.selection.rows[0]
    # Pega o ticker da linha selecionada (cuidado com a ordenação)
    ticker_val = df_view.sort_values('Liquidez', ascending=False).iloc[idx]['Ticker']
    # Busca a linha completa no DF original
    row = df_full[df_full['Ticker'] == ticker_val].iloc[0]
    sel_ticker = ticker_val

if sel_ticker:
    # Gera Tese
    bull, bear, tags, score = gerar_tese_investimento(row)
    
    st.markdown(f"## 🧠 Analista IA: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    # Topo do Painel
    col_kpi, col_chart = st.columns([1, 2])
    
    with col_kpi:
        # Score Card
        st.markdown(f"""
        <div style="background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; text-align: center;">
            <h1 style="color: {'#00ffbf' if score >= 7 else '#ffcc00' if score >= 4 else '#ff4444'}; margin: 0;">{score}/10</h1>
            <p style="color: #888;">Score de Qualidade</p>
            <hr style="border-color: #333;">
            <div style="display: flex; gap: 5px; justify-content: center; flex-wrap: wrap;">
                {''.join([f'<span class="badge-neutral" style="background: #333; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{t}</span>' for t in tags])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Valuation Card
        st.info(f"""
        **Valuation:**
        - Preço Atual: **R$ {row['Preco']:.2f}**
        - Graham (Justo): **R$ {row['Graham_Fair']:.2f}**
        - Upside Teórico: **{row['Upside']:.0f}%**
        """)

    with col_chart:
        # Chat Interface para a Tese
        with st.container(border=True):
            st.markdown("#### 📝 Tese de Investimento")
            
            # Mensagem do Analista
            with st.chat_message("assistant", avatar="🤖"):
                st.write(f"Aqui está minha análise sobre **{row['Ticker']} ({row['Setor']})**:")
                
                if bull:
                    st.markdown("**Pontos Fortes (Bull Case):**")
                    for p in bull: st.markdown(f"✅ {p}")
                
                if bear:
                    st.markdown("**Pontos de Atenção (Bear Case):**")
                    for p in bear: st.markdown(f"❌ {p}")
                    
                if not bull and not bear:
                    st.markdown("O ativo apresenta fundamentos neutros, sem grandes destaques ou riscos evidentes.")

    # Radar Chart e Detalhes
    c1, c2 = st.columns(2)
    with c1:
        # Normalização para Radar (0-5)
        def n(v, target): return min(5, max(0, (v/target)*5))
        
        radar_vals = [
            n(15/max(row['PL'], 1), 1), # Valuation (Inv)
            n(row['ROE'], 20),          # Qualidade
            n(row['DY'], 10),           # Renda
            n(1/max(row.get('Div_Patrimonio', 1), 0.1), 1), # Segurança (Inv)
            n(row.get('Cresc_5a', 0), 15) # Crescimento
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_vals,
            theta=['Valuation', 'Qualidade', 'Renda', 'Segurança', 'Crescimento'],
            fill='toself', name=sel_ticker,
            line_color='#00ffbf'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), template="plotly_dark", height=350, title="Radar de Fundamentos")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        # Gráfico Histórico (Yahoo On-Demand)
        try:
            with st.spinner("Carregando Gráfico..."):
                h = yf.download(sel_ticker+".SA", period="2y", progress=False)
                if not h.empty:
                    if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                    st.line_chart(h['Close'], color="#00ffbf", height=350)
                else: st.warning("Gráfico indisponível.")
        except: pass

else:
    st.info("👆 Clique em uma ação na tabela acima para gerar a Tese de Investimento com IA.")
