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
st.set_page_config(page_title="Titanium XXII | Institutional", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #000000; }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-family: 'DIN Alternate', 'Arial', sans-serif;
        font-size: 1.6rem;
        color: #00ffbf;
    }
    
    /* Tabela Comparativa Customizada */
    .comp-table {
        width: 100%;
        border-collapse: collapse;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 0.9rem;
    }
    .comp-table th { text-align: left; color: #888; padding: 5px; border-bottom: 1px solid #333; }
    .comp-table td { padding: 8px 5px; border-bottom: 1px solid #222; }
    .comp-val { font-weight: bold; }
    .comp-good { color: #4ade80; } /* Verde */
    .comp-bad { color: #f87171; }  /* Vermelho */
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #111; padding: 5px; border-radius: 6px; }
    .stTabs [data-baseweb="tab"] { height: 35px; border: none; color: #666; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #222 !important; color: #00ffbf !important; border-bottom: 2px solid #00ffbf; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Titanium XXII: Institutional Grade")

# --- 2. MOTOR DE DADOS BLINDADO ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_data_pro():
    url = 'https://www.fundamentus.com.br/resultado.php'
    # Headers completos para evitar erro "Sem Conexão"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status() # Verifica se deu erro 403/404
        
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Mapeamento Completo
        rename = {
            'Papel': 'Ticker', 'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'PSR': 'PSR',
            'Div.Yield': 'DY', 'P/Ativo': 'P_Ativo', 'P/Cap.Giro': 'P_CapGiro',
            'P/EBIT': 'P_EBIT', 'P/Ativ Circ Liq': 'P_AtivCircLiq',
            'EV/EBIT': 'EV_EBIT', 'EV/EBITDA': 'EV_EBITDA', 'Mrg Ebit': 'MargemEbit',
            'Mrg. Líq.': 'MargemLiquida', 'Liq. Corr.': 'LiqCorrente',
            'ROIC': 'ROIC', 'ROE': 'ROE', 'Liq.2meses': 'Liquidez',
            'Patrim. Líq': 'Patrimonio', 'Dív.Brut/ Patr.': 'Div_Patrimonio',
            'Cresc. Rec.5a': 'Cresc_5a', 'Giro Ativos': 'GiroAtivos'
        }
        
        # Filtra colunas que realmente vieram
        cols = [c for c in rename.keys() if c in df.columns]
        df = df[cols].rename(columns=rename)
        
        # Limpeza
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # Ajuste Percentual
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # Classificação Setorial (Manual Ampliada)
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA','BBSE','CXSE','IRBR']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN','FESA']: return 'Materiais Básicos'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR','RECV','ENAT']: return 'Petróleo & Gás'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA','ALPA','CVCB']: return 'Varejo Cíclico'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO','KEPL','SHUL','RAIL']: return 'Bens Industriais'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE','NEOE','AURE']: return 'Utilidade Pública'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL','ODPV','MATD','VVEO']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF','DIRR','CURY','TRIS']: return 'Construção'
            if t in ['ABEV','JBSS','BRFS','MRFG','BEEF','SMTO','MDIA','CRFB']: return 'Consumo Não Cíclico'
            if t in ['VIVT','TIMS','LWSA','TOTS','INTB','POSI']: return 'Tecnologia'
            if t in ['SLCE','AGRO','TTEN','SOJA']: return 'Agronegócio'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Garantia de Colunas
        req = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'PSR', 'LiqCorrente']
        for c in req: 
            if c not in df.columns: df[c] = 0.0
            
        # Score Institucional
        # Combina Valor + Qualidade + Momento (Liquidez)
        df['Inst_Score'] = (
            (df['PL'].rank(ascending=False, pct=True) * 0.3) +
            (df['ROE'].rank(ascending=True, pct=True) * 0.3) +
            (df['MargemLiquida'].rank(ascending=True, pct=True) * 0.2) +
            (df['Liquidez'].rank(ascending=True, pct=True) * 0.2)
        ) * 100

        return df
    except Exception as e:
        st.error(f"Erro Conexão B3: {e}")
        return pd.DataFrame()

# --- 3. FUNÇÃO DE COMPARAÇÃO HTML ---
def create_comparison_html(metric, val_acao, val_setor, better_is_higher=True, suffix=""):
    diff = val_acao - val_setor
    is_good = diff > 0 if better_is_higher else diff < 0
    color_class = "comp-good" if is_good else "comp-bad"
    arrow = "▲" if diff > 0 else "▼"
    
    return f"""
    <tr>
        <td>{metric}</td>
        <td class="comp-val">{val_acao:.2f}{suffix}</td>
        <td style="color:#888;">{val_setor:.2f}{suffix}</td>
        <td class="{color_class}">{arrow} {abs(diff):.2f}</td>
    </tr>
    """

# --- 4. INTERFACE ---
with st.spinner("Estabelecendo conexão segura..."):
    df_full = get_data_pro()

if df_full.empty:
    st.error("Falha crítica na fonte de dados. Tente recarregar.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Terminal Control")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor_list = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor_f = st.selectbox("Setor", setor_list)
    liq_f = st.select_slider("Liquidez Mínima", options=[0, 100000, 1000000, 5000000, 10000000], value=1000000)
    usar_yahoo = st.checkbox("Dados Trimestrais (Yahoo)", value=True)

# Filtro
mask = (df_full['Liquidez'] >= liq_f)
df_view = df_full[mask].copy()
if setor_f != "Todos": df_view = df_view[df_view['Setor'] == setor_f]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Layout
c1, c2 = st.columns([1.5, 2.5])

# Tabela Esquerda
sel_ticker = None
with c1:
    st.subheader(f"📋 Mercado ({len(df_view)})")
    
    cols_tab = ['Ticker', 'Preco', 'Inst_Score', 'PL', 'ROE', 'DY']
    cfg = {
        "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
        "Inst_Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
        "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
        "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
        "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%")
    }
    
    ev = st.dataframe(
        df_view[cols_tab].sort_values('Inst_Score', ascending=False),
        column_config=cfg,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=700
    )
    if len(ev.selection.rows) > 0:
        sel_ticker = df_view.sort_values('Inst_Score', ascending=False).iloc[ev.selection.rows[0]]['Ticker']

# Painel Direita
with c2:
    if sel_ticker:
        row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
        
        # Benchmarking
        peers = df_full[df_full['Setor'] == row['Setor']]
        # Mediana do setor (exclui outliers negativos/zeros para PL)
        med_pl = peers[(peers['PL']>0)&(peers['PL']<100)]['PL'].median()
        med_pvp = peers['PVP'].median()
        med_roe = peers['ROE'].median()
        med_dy = peers['DY'].median()
        med_mrg = peers['MargemLiquida'].median()
        med_div = peers['Div_Patrimonio'].median()
        
        st.markdown(f"## 🏛️ Raio-X Institucional: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
        st.caption(f"Setor: {row['Setor']} | Score: {row['Inst_Score']:.0f}/100")
        
        # 1. Comparativo Setorial (Tabela HTML)
        st.markdown("#### ⚖️ Comparativo vs. Pares (Setor)")
        html_table = f"""
        <table class="comp-table">
            <thead>
                <tr>
                    <th>Indicador</th>
                    <th>{sel_ticker}</th>
                    <th>Média Setor</th>
                    <th>Diferença</th>
                </tr>
            </thead>
            <tbody>
                {create_comparison_html("P/L (Anos)", row['PL'], med_pl, False, "x")}
                {create_comparison_html("P/VP", row['PVP'], med_pvp, False, "x")}
                {create_comparison_html("EV/EBIT", row['EV_EBIT'], peers['EV_EBIT'].median(), False, "x")}
                {create_comparison_html("ROE (Rentab.)", row['ROE'], med_roe, True, "%")}
                {create_comparison_html("Margem Líq.", row['MargemLiquida'], med_mrg, True, "%")}
                {create_comparison_html("Div. Yield", row['DY'], med_dy, True, "%")}
                {create_comparison_html("Dívida Líq/PL", row['Div_Patrimonio'], med_div, False, "x")}
                {create_comparison_html("Liquidez Corr.", row['LiqCorrente'], peers['LiqCorrente'].median(), True, "x")}
            </tbody>
        </table>
        """
        st.markdown(html_table, unsafe_allow_html=True)
        
        st.divider()
        
        # 2. Dados Avançados
        tab_g, tab_cont = st.tabs(["📈 Price Action & Tendência", "📑 Contabilidade Trimestral (QoQ)"])
        
        with tab_g:
            if usar_yahoo:
                try:
                    with st.spinner("Baixando Histórico..."):
                        h = yf.download(sel_ticker+".SA", period="2y", progress=False)
                        if not h.empty:
                            if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                            
                            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
                            fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, title="Gráfico Diário (2 Anos)")
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.warning("Gráfico indisponível.")
                except: st.error("Erro no gráfico.")
            else: st.info("Ative Yahoo para gráficos.")
            
        with tab_cont:
            if usar_yahoo:
                try:
                    stock = yf.Ticker(sel_ticker+".SA")
                    q = stock.quarterly_financials.T.sort_index(ascending=True)
                    if not q.empty and len(q) > 1:
                        # Seleciona colunas chave
                        cols_want = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
                        q_clean = q[ [c for c in cols_want if c in q.columns] ].copy()
                        
                        # Tradução
                        q_clean.columns = ['Receita', 'Lucro Bruto', 'EBIT', 'Lucro Líquido']
                        
                        # Variação %
                        q_pct = q_clean.pct_change() * 100
                        q_pct = q_pct.add_suffix(" (QoQ %)")
                        
                        st.markdown("**Evolução Trimestral (R$):**")
                        st.dataframe(q_clean.style.format("{:,.0f}"), use_container_width=True)
                        
                        # Gráfico Margens
                        q_clean['Margem Bruta'] = (q_clean['Lucro Bruto'] / q_clean['Receita']) * 100
                        q_clean['Margem Líquida'] = (q_clean['Lucro Líquido'] / q_clean['Receita']) * 100
                        
                        fig_m = px.line(q_clean, y=['Margem Bruta', 'Margem Líquida'], markers=True, title="Tendência de Margens (%)", template="plotly_dark")
                        st.plotly_chart(fig_m, use_container_width=True)
                        
                    else: st.warning("Dados trimestrais indisponíveis.")
                except: st.error("Erro contábil.")
    
    else:
        st.info("👈 Selecione um ativo para ver o Comparativo Setorial.")
