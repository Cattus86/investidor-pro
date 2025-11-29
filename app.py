import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO DE TERMINAL (VISUAL ELITE) ---
st.set_page_config(page_title="Titanium Pro XIII", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Fundo Dark Profissional */
    .stApp { background-color: #0e1117; }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.4rem;
        color: #00ffbf;
        font-weight: 700;
    }
    
    /* Tabelas */
    div[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
    
    /* Cards de Análise */
    .ai-box {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00ffbf;
        margin-bottom: 10px;
    }
    .ai-good { color: #4caf50; font-weight: bold; }
    .ai-bad { color: #f44336; font-weight: bold; }
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; background-color: #161b22; padding: 5px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { height: 35px; border: none; color: #8b949e; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro XIII: Executive Terminal")

# --- 2. MOTOR DE DADOS ROBUSTO ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

def normalize_cols(cols):
    new_cols = []
    for col in cols:
        nfkd = unicodedata.normalize('NFKD', col)
        c = "".join([c for c in nfkd if not unicodedata.combining(c)])
        c = c.replace('.', '').replace('/', '').replace(' ', '').lower()
        new_cols.append(c)
    return new_cols

@st.cache_data(ttl=600, show_spinner=False)
def get_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers)
        dfs = pd.read_html(StringIO(r.text), decimal=',', thousands='.')
        if not dfs: return pd.DataFrame()
        df = dfs[0]
        
        # 1. Normalização
        df.columns = normalize_cols(df.columns)
        
        # 2. Mapeamento
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'pativo': 'P_Ativo', 'pcapgiro': 'P_CapGiro',
            'pebit': 'P_EBIT', 'evebit': 'EV_EBIT', 'evebitda': 'EV_EBITDA', 
            'mrgebit': 'MargemEbit', 'mrgliq': 'MargemLiquida', 'liqcorr': 'LiqCorrente',
            'roic': 'ROIC', 'roe': 'ROE', 'liq2meses': 'Liquidez',
            'patrimliq': 'Patrimonio', 'divbrutpatr': 'Div_Patrimonio',
            'crescrec5a': 'Cresc_5a'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # 3. Limpeza Numérica
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # 4. Ajuste Percentual
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # 5. Garantia de Colunas
        req_cols = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'PSR', 'LiqCorrente']
        for c in req_cols:
            if c not in df.columns: df[c] = 0.0

        # 6. Setores
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE','NEOE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            return 'Outros'
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # 7. Rankings
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0)
        
        # Quality Score (0-100)
        df['Quality'] = (
            (df['ROE'].rank(pct=True)*40) + 
            (df['MargemLiquida'].rank(pct=True)*30) + 
            ((df['Div_Patrimonio']*-1).rank(pct=True)*30)
        )

        return df
    except Exception as e:
        st.error(f"Erro de conexão: {e}")
        return pd.DataFrame()

# --- 3. INTELIGÊNCIA ARTIFICIAL (LÓGICA) ---
def gerar_relatorio_ia(row):
    """Gera análise textual e dados para o gráfico de Radar"""
    pos = []
    neg = []
    
    # Lógica de Análise
    if row['PL'] > 0 and row['PL'] < 10: pos.append(f"Valuation atrativo (P/L {row['PL']:.1f}x).")
    elif row['PL'] > 25: neg.append("Múltiplo de lucro esticado.")
    
    if row['PVP'] < 1: pos.append("Negociada abaixo do valor patrimonial.")
    
    if row['ROE'] > 15: pos.append(f"Alta eficiência (ROE {row['ROE']:.1f}%).")
    elif row['ROE'] < 5: neg.append("Rentabilidade baixa.")
    
    if row['Div_Patrimonio'] < 0.8: pos.append("Baixo endividamento.")
    elif row['Div_Patrimonio'] > 3: neg.append(f"Alavancagem alta ({row['Div_Patrimonio']:.1f}x).")
    
    if row['DY'] > 6: pos.append(f"Bom pagador de dividendos ({row['DY']:.1f}%).")
    
    # Score Radar (Normalizado 0-5)
    def n(val, best, worst):
        return max(0, min(5, ((val - worst) / (best - worst)) * 5))
    
    radar_vals = [
        n(15/max(row['PL'], 1), 2, 0),      # Valuation (Inv)
        n(row['ROE'], 25, 0),               # Qualidade
        n(row['DY'], 12, 0),                # Renda
        n(row['Cresc_5a'], 20, -10),        # Crescimento
        n(1/max(row['Div_Patrimonio'],0.1), 2, 0) # Segurança (Inv)
    ]
    
    return pos, neg, radar_vals

# --- 4. INTERFACE ---
with st.spinner("Conectando ao Mercado..."):
    df_full = get_data()

if df_full.empty:
    st.error("Sistema offline. Tente recarregar.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor = st.selectbox("Setor", ["Todos"] + sorted(df_full['Setor'].unique().tolist()))
    
    with st.expander("📊 Indicadores", expanded=True):
        liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 1000000, 5000000], value=500000)
        pl_r = st.slider("P/L", -10.0, 50.0, (-5.0, 30.0))
        dy_r = st.slider("DY %", 0.0, 20.0, (0.0, 20.0))
        roe_m = st.slider("ROE Min", -20.0, 40.0, 0.0)

# Filtros
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
c2.metric("Média P/L", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
c3.metric("Média DY", f"{df_view['DY'].mean():.1f}%")

t_main = st.tabs(["Geral", "💰 Dividendos", "💎 Valor", "🛡️ Qualidade", "🚀 Crescimento"])

cols_main = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Graham_Fair', 'Upside', 'Quality']

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%"),
    "Div_Patrimonio": st.column_config.NumberColumn("Dívida/PL", format="%.2f"),
    "Graham_Fair": st.column_config.NumberColumn("Justo", format="R$ %.2f"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Quality": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
    "Cresc_5a": st.column_config.NumberColumn("CAGR 5a", format="%.1f%%")
}

sel_ticker = None

def render_table(df_in, key):
    safe_cols = [c for c in cols_main if c in df_in.columns]
    ev = st.dataframe(df_in[safe_cols], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=400, key=key)
    if len(ev.selection.rows) > 0: return df_in.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t_main[0]: sel_ticker = render_table(df_view.sort_values('Liquidez', ascending=False), 't1')
with t_main[1]: sel_ticker = render_table(df_view.nlargest(100, 'DY'), 't2')
with t_main[2]: sel_ticker = render_table(df_view[(df_view['Upside']>0)].nlargest(100, 'Upside'), 't3')
with t_main[3]: sel_ticker = render_table(df_view.nlargest(100, 'Quality'), 't4')
with t_main[4]: sel_ticker = render_table(df_view.nlargest(100, 'Cresc_5a'), 't5')

# --- PAINEL DE ANÁLISE ---
st.divider()

if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    pos, neg, radar_data = gerar_relatorio_ia(row)
    
    st.markdown(f"## 🤖 Analista Virtual: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    col_kpi, col_radar, col_ia = st.columns([1, 1.5, 1.5])
    
    with col_kpi:
        st.metric("Preço", f"R$ {row['Preco']:.2f}")
        st.metric("P/L", f"{row['PL']:.1f}x")
        st.metric("P/VP", f"{row['PVP']:.2f}x")
        st.metric("ROE", f"{row['ROE']:.1f}%")
        
    with col_radar:
        # GRÁFICO DE RADAR (SPIDER)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_data,
            theta=['Valuation', 'Qualidade', 'Renda', 'Crescimento', 'Segurança'],
            fill='toself', name=sel_ticker,
            line_color='#00ffbf'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            template="plotly_dark",
            title="Raio-X de Fundamentos (0-5)",
            margin=dict(t=30, b=20, l=40, r=40),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_ia:
        st.markdown('<div class="ai-box">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Opinião da IA")
        
        if pos:
            st.markdown('<span class="ai-good">DESTAQUES POSITIVOS:</span>', unsafe_allow_html=True)
            for p in pos: st.markdown(f"✅ {p}")
        else: st.markdown("⚠️ Nenhum destaque positivo óbvio.")
            
        st.markdown("---")
        
        if neg:
            st.markdown('<span class="ai-bad">PONTOS DE ATENÇÃO:</span>', unsafe_allow_html=True)
            for n in neg: st.markdown(f"❌ {n}")
        else: st.markdown("🛡️ Nenhum alerta crítico detectado.")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Detalhes Extras
    with st.expander(f"📈 Ver Gráfico Histórico de {sel_ticker} (Requer conexão Yahoo)", expanded=False):
        try:
            with st.spinner("Baixando..."):
                h = yf.download(sel_ticker+".SA", period="2y", progress=False)
                if not h.empty:
                    if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                    st.line_chart(h['Close'])
        except: st.error("Gráfico indisponível.")

else:
    st.info("👆 Selecione um ativo na tabela para gerar o relatório.")
