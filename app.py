import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO DE TERMINAL (BLACK OBSIDIAN THEME) ---
st.set_page_config(page_title="Titanium XX | The Monolith", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Fundo Absoluto */
    .stApp { background-color: #000000; }
    
    /* Métricas Monolíticas */
    [data-testid="stMetricValue"] {
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 2rem;
        color: #00ffcc;
        font-weight: 800;
        text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.4);
    }
    
    /* Tabela Profissional */
    div[data-testid="stDataFrame"] div[class*="stDataFrame"] { border: 1px solid #222; }
    
    /* Card do Analista */
    .analyst-card {
        background: linear-gradient(145deg, #0a0a0a, #111);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    /* Badges */
    .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; margin-right: 5px; }
    .b-green { background: #064025; color: #4ade80; border: 1px solid #4ade80; }
    .b-red { background: #400606; color: #f87171; border: 1px solid #f87171; }
    .b-blue { background: #062640; color: #60a5fa; border: 1px solid #60a5fa; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Titanium XX: The Monolith")

# --- 2. MOTOR DE DADOS & SCORING SYSTEM ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_monolith_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # 1. Mapeamento Total
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
        
        # Normaliza colunas do HTML antes de renomear
        df.columns = [c.replace('.', '').replace('/', '').replace(' ', '').lower() for c in df.columns]
        # Mapa reverso baseado na normalização (ajuste técnico)
        rev_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'evebit': 'EV_EBIT', 'roic': 'ROIC', 'roe': 'ROE',
            'liq2meses': 'Liquidez', 'mrgliq': 'MargemLiquida', 'mrgebit': 'MargemEbit',
            'divbrutpatr': 'Div_Patrimonio', 'crescrec5a': 'Cresc_5a', 'liqcorr': 'LiqCorrente'
        }
        
        cols = [c for c in rev_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rev_map)
        
        # Limpeza
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # Percentuais
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns:
                if df[col].mean() < 1: df[col] *= 100
        
        # Colunas de Garantia
        cols_req = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'PSR', 'LiqCorrente']
        for c in cols_req: 
            if c not in df.columns: df[c] = 0.0

        # Setor
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN']: return 'Energia'
            if t in ['MGLU','LREN','ARZZ','PETZ']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY']: return 'Saúde'
            return 'Geral'
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # --- ALGORITMO DE SCORE GLOBAL (0-100) ---
        # 1. Rank Valuation (Menor PL/PVP é melhor)
        # Invertemos o rank para que menor PL dê maior pontuação
        rank_val = (df['PL'].rank(ascending=False, pct=True) + df['PVP'].rank(ascending=False, pct=True)) / 2
        
        # 2. Rank Qualidade (Maior ROE/Margem é melhor)
        rank_qual = (df['ROE'].rank(ascending=True, pct=True) + df['MargemLiquida'].rank(ascending=True, pct=True)) / 2
        
        # 3. Rank Renda (Maior DY é melhor)
        rank_inc = df['DY'].rank(ascending=True, pct=True)
        
        # 4. Rank Segurança (Menor Dívida é melhor)
        rank_safe = (df['Div_Patrimonio'] * -1).rank(ascending=True, pct=True)
        
        # Peso do Score Final
        # Valuation 30%, Qualidade 30%, Renda 20%, Segurança 20%
        df['Global_Score'] = (rank_val * 30) + (rank_qual * 30) + (rank_inc * 20) + (rank_safe * 20)
        
        # Cálculos Auxiliares
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham']>0), ((df['Graham']-df['Preco'])/df['Preco'])*100, -999)
        
        return df
    except Exception as e:
        st.error(f"Erro: {e}")
        return pd.DataFrame()

# --- 3. CÉREBRO ANALISTA SUPER POTENTE ---
def super_analista(row):
    score = row['Global_Score']
    
    # Classificação
    if score >= 80: veredicto = "COMPRA FORTE (Strong Buy)"
    elif score >= 60: veredicto = "COMPRA (Buy)"
    elif score >= 40: veredicto = "NEUTRO (Hold)"
    else: veredicto = "VENDA (Sell)"
    
    insights = []
    risks = []
    
    # Análise Profunda
    if row['Graham'] > row['Preco'] * 1.5:
        insights.append(f"💎 **Margem de Segurança:** Negociada com desconto massivo de {row['Upside']:.0f}% sobre o valor intrínseco de Graham.")
    
    if row['ROE'] > 20 and row['PL'] < 10:
        insights.append("🚀 **Qualidade Barata:** Combinação rara de alta rentabilidade (ROE > 20%) com múltiplos baixos (P/L < 10).")
        
    if row['LiqCorrente'] > 2:
        insights.append("🛡️ **Caixa Forte:** Alta liquidez corrente, empresa preparada para crises.")
        
    if row['Div_Patrimonio'] > 3:
        risks.append(f"⚠️ **Alavancagem:** Dívida 3x maior que o patrimônio. Atenção aos juros.")
        
    if row['MargemLiquida'] < 5:
        risks.append("⚠️ **Margens Apertadas:** Baixa eficiência operacional, sensível a custos.")
        
    return veredicto, score, insights, risks

# --- 4. INTERFACE ---
with st.spinner("Carregando o Monolito..."):
    df_full = get_monolith_data()

if df_full.empty:
    st.stop()

# --- SIDEBAR COMPACTA ---
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor = st.selectbox("Setor", ["Todos"] + sorted(df_full['Setor'].unique().tolist()))
    liq_min = st.select_slider("Liquidez", options=[0, 500000, 1000000, 5000000, 10000000], value=1000000)
    
    with st.expander("Filtros Avançados"):
        pl_min, pl_max = st.slider("P/L", -5.0, 50.0, (-5.0, 30.0))
        roe_min = st.slider("ROE Min", -20.0, 40.0, 0.0)

# FILTRAGEM
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_min, pl_max)) &
    (df_full['ROE'] >= roe_min)
)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# --- LAYOUT PRINCIPAL ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ativos no Radar", len(df_view))
c2.metric("Score Médio", f"{df_view['Global_Score'].mean():.0f}/100")
top_stock = df_view.sort_values('Global_Score', ascending=False).iloc[0]
c3.metric("Top Pick", top_stock['Ticker'])
c4.metric("Upside Top Pick", f"{top_stock['Upside']:.0f}%")

st.divider()

# --- TABELA ÚNICA PODEROSA ---
st.subheader(f"📋 Ranking Unificado (Global Quant Score)")

# Ordenação Padrão: Global Score
df_show = df_view.sort_values('Global_Score', ascending=False)

# Colunas para exibir
cols_show = ['Ticker', 'Global_Score', 'Preco', 'Graham', 'Upside', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Liquidez']

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "Graham": st.column_config.NumberColumn("Justo", format="R$ %.2f"),
    "Global_Score": st.column_config.ProgressColumn("🏆 Score", format="%.0f", min_value=0, max_value=100),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "Liquidez": st.column_config.NumberColumn("Liq.", format="%.0e")
}

ev = st.dataframe(
    df_show[cols_show],
    column_config=cfg,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    height=400
)

# --- ANALISTA SUPER POTENTE ---
st.divider()

sel_ticker = None
if len(ev.selection.rows) > 0:
    sel_ticker = df_show.iloc[ev.selection.rows[0]]['Ticker']

if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    veredito, score, insights, risks = super_analista(row)
    
    st.markdown(f"## 🏛️ Análise Institucional: <span style='color:#00ffcc'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    # Layout Analista
    c_left, c_right = st.columns([1.5, 1])
    
    with c_left:
        # Card Principal
        color_score = "#00ffcc" if score >= 80 else "#ffcc00" if score >= 50 else "#ff4444"
        
        st.markdown(f"""
        <div class="analyst-card" style="border-left: 5px solid {color_score};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin:0; color: #888;">VEREDICTO QUANTITATIVO</h4>
                    <h1 style="margin:0; color: white;">{veredito}</h1>
                </div>
                <div style="text-align: right;">
                    <h1 style="margin:0; font-size: 3.5rem; color: {color_score};">{score:.0f}</h1>
                    <small style="color: #888;">Global Score</small>
                </div>
            </div>
            <hr style="border-color: #333;">
            <div style="margin-top: 15px;">
                <h4 style="color: #00ffcc;">🚀 Teses de Alta (Bull Case)</h4>
                {''.join([f'<p>✅ {i}</p>' for i in insights]) if insights else "<p style='color:#666'>Nenhum destaque positivo relevante.</p>"}
            </div>
            <div style="margin-top: 15px;">
                <h4 style="color: #ff4444;">⚠️ Riscos (Bear Case)</h4>
                {''.join([f'<p>🔻 {r}</p>' for r in risks]) if risks else "<p style='color:#666'>Nenhum risco crítico detectado.</p>"}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico Histórico (Yahoo On Demand)
        with st.expander("📈 Ver Gráfico de Preço (5 Anos)", expanded=True):
            try:
                with st.spinner("Baixando..."):
                    h = yf.download(sel_ticker+".SA", period="5y", progress=False)
                    if not h.empty:
                        if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                        st.line_chart(h['Close'], color="#00ffcc", height=250)
            except: st.error("Gráfico indisponível.")

    with c_right:
        # Radar Chart
        st.markdown("#### 🧭 Bússola de Fundamentos")
        
        # Normalização 0-100 para o radar
        r_vals = [
            min(100, (15/max(row['PL'],1))*100), # Valor
            min(100, (row['ROE']/25)*100),       # Qualidade
            min(100, (row['DY']/12)*100),        # Renda
            min(100, (1/max(row['Div_Patrimonio'],0.1))*100), # Segurança
            min(100, (row['MargemLiquida']/20)*100) # Eficiência
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=['Valor', 'Qualidade', 'Renda', 'Segurança', 'Eficiência'],
            fill='toself', name=sel_ticker,
            line_color='#00ffcc', fillcolor='rgba(0, 255, 204, 0.2)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], color='#444')),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=40, r=40),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # KPIs Grid
        c1, c2 = st.columns(2)
        c1.metric("P/L", f"{row['PL']:.1f}x")
        c2.metric("P/VP", f"{row['PVP']:.2f}x")
        c1.metric("ROE", f"{row['ROE']:.1f}%")
        c2.metric("Dívida/PL", f"{row.get('Div_Patrimonio',0):.2f}")

else:
    st.info("👆 Clique em um ativo na Tabela Unificada para ativar o Analista.")
