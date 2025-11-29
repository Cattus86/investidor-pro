import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO VISUAL INSTITUCIONAL ---
st.set_page_config(page_title="Titanium XXI | Hedge Fund", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #000000; }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-family: 'Consolas', monospace;
        font-size: 1.5rem;
        color: #00e676;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] { font-size: 0.9rem; color: #888; }
    
    /* Tabelas */
    div[data-testid="stDataFrame"] { border: 1px solid #333; }
    
    /* Box de Comparação */
    .comp-box {
        background-color: #111;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .bull { color: #4ade80; font-weight: bold; }
    .bear { color: #f87171; font-weight: bold; }
    .neutral { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Titanium XXI: Hedge Fund Manager")

# --- 2. MOTOR DE DADOS & CLEANING ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Renomeia colunas para facilitar
        rename = {
            'Papel': 'Ticker', 'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 
            'Div.Yield': 'DY', 'ROE': 'ROE', 'Margem Líquida': 'MargemLiquida', 
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a',
            'Liq. Corr.': 'LiqCorrente', 'Liq.2meses': 'Liquidez',
            'EV/EBIT': 'EV_EBIT', 'ROIC': 'ROIC'
        }
        
        # Normaliza nomes do HTML
        df.columns = [c.replace('.', '').replace('/', '').replace(' ', '') if c not in rename else c for c in df.columns]
        # Aplica renomeação manual
        col_map = {'Papel':'Ticker', 'Cotacao':'Preco', 'PL':'PL', 'PVP':'PVP', 'DivYield':'DY', 'ROE':'ROE', 'MrgLiq':'MargemLiquida', 'DivBrutPatr':'Div_Patrimonio', 'CrescRec5a':'Cresc_5a', 'LiqCorr':'LiqCorrente', 'Liq2meses':'Liquidez', 'EVEBIT':'EV_EBIT', 'ROIC':'ROIC'}
        df.rename(columns=col_map, inplace=True)
        
        # Limpeza
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # Percentuais
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # Classificação Setorial (Manual Robusta)
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN']: return 'Materiais Básicos'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo & Gás'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO','KEPL']: return 'Bens Industriais'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Utilidade Pública'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF','DIRR']: return 'Construção'
            if t in ['SLCE','AGRO','TTEN','SOJA']: return 'Agronegócio'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Score Básico para Tabela
        df['Score'] = (df['ROE'].rank(pct=True) + df['DY'].rank(pct=True) + (df['PL']*-1).rank(pct=True))*33
        
        return df
    except: return pd.DataFrame()

# --- 3. MOTOR TRIMESTRAL (YAHOO) ---
def get_quarterly_data(ticker):
    try:
        stock = yf.Ticker(ticker+".SA")
        # Pega trimestral
        q_inc = stock.quarterly_financials.T.sort_index(ascending=True)
        
        if q_inc.empty or len(q_inc) < 2: return None
        
        # Filtra colunas essenciais
        cols_map = {'Total Revenue': 'Receita', 'Net Income': 'Lucro Líquido', 'EBITDA': 'EBITDA'}
        df_q = pd.DataFrame(index=q_inc.index)
        
        for k, v in cols_map.items():
            if k in q_inc.columns:
                df_q[v] = q_inc[k]
        
        # Cálculos de Variação (QoQ)
        df_q['Receita QoQ %'] = df_q['Receita'].pct_change() * 100
        df_q['Lucro QoQ %'] = df_q['Lucro Líquido'].pct_change() * 100
        
        return df_q.iloc[-5:] # Últimos 5 trimestres
    except: return None

# --- 4. MOTOR DE COMPARAÇÃO SETORIAL ---
def get_sector_benchmarks(df_full, setor):
    peers = df_full[df_full['Setor'] == setor]
    
    # Calcula Medianas (Melhor que média para evitar distorções)
    bench = {
        'PL': peers[(peers['PL']>0)&(peers['PL']<100)]['PL'].median(),
        'ROE': peers['ROE'].median(),
        'DY': peers['DY'].median(),
        'MargemLiquida': peers['MargemLiquida'].median(),
        'Div_Patrimonio': peers['Div_Patrimonio'].median()
    }
    return bench

# --- 5. INTERFACE ---
with st.spinner("Carregando Terminal Institucional..."):
    df_full = get_data()

if df_full.empty:
    st.error("Sem conexão.")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor_f = st.selectbox("Setor", ["Todos"] + sorted(df_full['Setor'].unique().tolist()))
    liq_f = st.select_slider("Liquidez", options=[0, 100000, 1000000, 5000000], value=1000000)
    
    usar_yahoo = st.checkbox("Ativar Dados Trimestrais", value=True)

# FILTER
mask = (df_full['Liquidez'] >= liq_f)
df_view = df_full[mask].copy()
if setor_f != "Todos": df_view = df_view[df_view['Setor'] == setor_f]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# LAYOUT
c1, c2 = st.columns([1.2, 2.5])

# TABELA
sel_ticker = None
with c1:
    st.subheader(f"📋 Screener ({len(df_view)})")
    
    # Colunas seguras
    cols_show = ['Ticker', 'Preco', 'PL', 'ROE', 'DY', 'Score']
    cols_safe = [c for c in cols_show if c in df_view.columns]
    
    cfg = {
        "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
        "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
        "DY": st.column_config.ProgressColumn("DY", format="%.1f%%", min_value=0, max_value=15),
        "Score": st.column_config.ProgressColumn("Rank", min_value=0, max_value=100),
    }
    
    ev = st.dataframe(
        df_view[cols_safe].sort_values('Score', ascending=False),
        column_config=cfg,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=700
    )
    if len(ev.selection.rows) > 0:
        sel_ticker = df_view.sort_values('Score', ascending=False).iloc[ev.selection.rows[0]]['Ticker']

# PAINEL ANALISTA
with c2:
    if sel_ticker:
        row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
        bench = get_sector_benchmarks(df_full, row['Setor'])
        
        st.markdown(f"## 🏛️ Hedge Fund Report: <span style='color:#00e676'>{sel_ticker}</span>", unsafe_allow_html=True)
        st.caption(f"Setor: {row['Setor']} | Liquidez: R$ {row['Liquidez']/1e6:.1f}M")
        
        # 1. VALUATION RELATIVO (AÇÃO vs SETOR)
        st.markdown("#### ⚖️ Valuation Relativo (vs Pares)")
        
        k1, k2, k3, k4 = st.columns(4)
        
        # Lógica de Cor: Se for melhor que o setor, Verde. Pior, Vermelho.
        # PL (Menor é melhor)
        pl_delta = ((row['PL'] - bench['PL']) / bench['PL']) * 100
        pl_color = "normal" if abs(pl_delta) < 10 else "inverse" # Streamlit delta logic
        k1.metric("P/L Ação", f"{row['PL']:.1f}x", f"{pl_delta:.1f}% vs Setor", delta_color="inverse")
        
        # ROE (Maior é melhor)
        roe_delta = row['ROE'] - bench['ROE']
        k2.metric("ROE Ação", f"{row['ROE']:.1f}%", f"{roe_delta:.1f}pp vs Setor")
        
        # DY (Maior é melhor)
        dy_delta = row['DY'] - bench['DY']
        k3.metric("Yield Ação", f"{row['DY']:.1f}%", f"{dy_delta:.1f}pp vs Setor")
        
        # Margem
        mrg_delta = row['MargemLiquida'] - bench['MargemLiquida']
        k4.metric("Margem Líq.", f"{row['MargemLiquida']:.1f}%", f"{mrg_delta:.1f}pp vs Setor")
        
        st.markdown("---")
        
        # 2. ANÁLISE TRIMESTRAL (QoQ)
        st.markdown("#### 📉 Performance Trimestral (Evolução Recente)")
        
        if usar_yahoo:
            with st.spinner("Baixando Balanços Trimestrais..."):
                df_q = get_quarterly_data(sel_ticker)
                
                if df_q is not None:
                    # Gráfico Combinado (Barra Receita + Linha Lucro)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Receita', x=df_q.index, y=df_q['Receita'], marker_color='#1f77b4'))
                    fig.add_trace(go.Scatter(name='Lucro Líquido', x=df_q.index, y=df_q['Lucro Líquido'], yaxis='y2', line=dict(color='#00e676', width=3)))
                    
                    fig.update_layout(
                        title="Receita (Barras) vs Lucro (Linha) - Últimos Trimestres",
                        yaxis=dict(title="Receita"),
                        yaxis2=dict(title="Lucro", overlaying='y', side='right'),
                        template="plotly_dark",
                        legend=dict(orientation="h", y=1.1),
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela de Variação
                    st.markdown("**Variação Trimestral (QoQ %):**")
                    cols_show = ['Receita', 'Receita QoQ %', 'Lucro Líquido', 'Lucro QoQ %']
                    st.dataframe(
                        df_q[cols_show].style.format("{:,.2f}").background_gradient(subset=['Receita QoQ %', 'Lucro QoQ %'], cmap='RdYlGn', vmin=-20, vmax=20),
                        use_container_width=True
                    )
                    
                    # Insights Automáticos
                    last_q = df_q.iloc[0] # Yahoo ordena o mais recente primeiro (ou ultimo index dependendo do sort)
                    # No nosso código fiz sort_index(ascending=True), então o último é o mais recente.
                    last_q = df_q.iloc[-1]
                    
                    txt_analise = []
                    if last_q['Receita QoQ %'] > 0 and last_q['Lucro QoQ %'] > 0:
                        txt_analise.append("🟢 **Trimestre Forte:** Empresa cresceu tanto Receita quanto Lucro no último tri.")
                    elif last_q['Receita QoQ %'] > 0 and last_q['Lucro QoQ %'] < 0:
                        txt_analise.append("🟡 **Compressão:** Receita subiu, mas Lucro caiu. Custos aumentaram?")
                    elif last_q['Receita QoQ %'] < 0:
                        txt_analise.append("🔴 **Desaceleração:** Receita caiu frente ao trimestre anterior.")
                        
                    st.info(" ".join(txt_analise))
                    
                else:
                    st.warning("Dados trimestrais não disponíveis no Yahoo Finance para este ativo.")
        else:
            st.info("Ative a opção Yahoo para ver dados trimestrais.")
            
        st.markdown("---")
        
        # 3. VEREDITO COMPARATIVO
        c_radar, c_txt = st.columns([1, 1.5])
        
        with c_radar:
            # Radar: Ação vs Setor (Normalizado)
            # Precisamos normalizar para escala 0-100 para o gráfico fazer sentido
            # Usaremos: (Valor Ação / Valor Setor) * 50. Se igual = 50. Se dobro = 100.
            
            r_pl = 50 * (bench['PL'] / max(row['PL'], 0.1)) # PL menor é melhor
            r_roe = 50 * (row['ROE'] / max(bench['ROE'], 0.1))
            r_dy = 50 * (row['DY'] / max(bench['DY'], 0.1))
            r_div = 50 * (bench['Div_Patrimonio'] / max(row.get('Div_Patrimonio', 1), 0.1))
            
            # Limita a 100
            vals = [min(100, x) for x in [r_pl, r_roe, r_dy, r_div]]
            
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=['Valuation (P/L)', 'Rentabilidade (ROE)', 'Dividendos', 'Segurança (Dívida)'],
                fill='toself', name='Ação', line_color='#00ffcc'
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=[50, 50, 50, 50], theta=['Valuation (P/L)', 'Rentabilidade (ROE)', 'Dividendos', 'Segurança (Dívida)'],
                name='Média Setor', line_color='#666', line_dash='dash'
            ))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), template="plotly_dark", height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_r, use_container_width=True)
            
        with c_txt:
            st.markdown("#### 🏆 Conclusão Setorial")
            if row['ROE'] > bench['ROE'] and row['PL'] < bench['PL']:
                st.success(f"**JOIA RARA:** {sel_ticker} é mais rentável (ROE {row['ROE']}%) e mais barata (P/L {row['PL']}x) que a média do setor.")
            elif row['ROE'] < bench['ROE'] and row['PL'] > bench['PL']:
                st.error(f"**CARA E PIOR:** {sel_ticker} tem rentabilidade menor e custa mais caro que os pares.")
            else:
                st.warning(f"**EM LINHA:** Ação negociada dentro dos parâmetros justos do setor.")

    else:
        st.info("👆 Selecione um ativo para iniciar a análise Hedge Fund.")
        
        # Market Map
        try:
            df_map = df_view.groupby('Setor')[['Liquidez', 'ROE']].median().reset_index()
            df_map['Qtd'] = df_view.groupby('Setor')['Ticker'].count().values
            fig = px.treemap(df_map, path=['Setor'], values='Qtd', color='ROE', color_continuous_scale='Viridis', title="Mapa de Calor: Setores por ROE")
            st.plotly_chart(fig, use_container_width=True)
        except: pass
