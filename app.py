import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO DE TERMINAL ---
st.set_page_config(page_title="Titanium Pro X", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    [data-testid="stMetricValue"] { font-size: 1.3rem; color: #00ffbf; font-family: 'Roboto Mono', monospace; font-weight: 700; }
    .stDataFrame { border: 1px solid #30363d; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #161b22; padding: 5px; border-radius: 6px; }
    .stTabs [data-baseweb="tab"] { height: 35px; border: none; color: #8b949e; font-weight: 600; font-size: 13px; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro X: Anti-Crash Edition")

# --- 2. FUNÇÕES DE DADOS ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

def normalize_cols(cols):
    """Remove acentos, espaços e pontos para padronizar nomes"""
    new_cols = []
    for col in cols:
        # Remove acentos
        nfkd_form = unicodedata.normalize('NFKD', col)
        col = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        # Remove especiais e lower
        col = col.replace('.', '').replace('/', '').replace(' ', '').lower()
        new_cols.append(col)
    return new_cols

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers)
        # Tenta ler a tabela. Se falhar, retorna vazio.
        dfs = pd.read_html(StringIO(r.text), decimal=',', thousands='.')
        if not dfs: return pd.DataFrame()
        df = dfs[0]
        
        # 1. Normalização dos Nomes das Colunas (A Correção do Erro)
        # Ex: "Dív.Brut/ Patr." vira "divbrutpatr"
        df.columns = normalize_cols(df.columns)
        
        # 2. Mapeamento Seguro
        # Mapeia do nome normalizado -> Nome Bonito Interno
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'pativo': 'P_Ativo', 'pcapgiro': 'P_CapGiro',
            'pebit': 'P_EBIT', 'pativcircliq': 'P_AtivCircLiq',
            'evebit': 'EV_EBIT', 'evebitda': 'EV_EBITDA', 'mrgebit': 'MargemEbit',
            'mrgliq': 'MargemLiquida', 'liqcorr': 'LiqCorrente',
            'roic': 'ROIC', 'roe': 'ROE', 'liq2meses': 'Liquidez',
            'patrimliq': 'Patrimonio', 'divbrutpatr': 'Div_Patrimonio', # Aqui estava o erro
            'crescrec5a': 'Cresc_5a'
        }
        
        df.rename(columns=rename_map, inplace=True)
        
        # 3. Limpeza de Dados
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # 4. Ajuste Percentual
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # 5. Garantia de Colunas (Se faltar alguma, cria zerada)
        required_cols = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a']
        for c in required_cols:
            if c not in df.columns: df[c] = 0.0

        # 6. Setores
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # 7. Rankings (Agora seguro porque as colunas existem)
        df['Graham_Fair'] = np.where((df['PL']>0)&(df['PVP']>0), np.sqrt(22.5 * (df['Preco']/df['PL']) * (df['Preco']/df['PVP'])), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0)
        
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999
            
        # Quality Score
        df['Quality_Score'] = (df['ROE'].rank(ascending=True) + df['MargemLiquida'].rank(ascending=True) + (df['Div_Patrimonio'] * -1).rank(ascending=True))

        return df
    except Exception as e:
        st.error(f"Erro no processamento de dados: {e}")
        return pd.DataFrame()

# --- 3. CONTABILIDADE ---
def get_accounting_matrix(ticker):
    try:
        stock = yf.Ticker(ticker+".SA")
        inc = stock.financials.T.sort_index(ascending=True)
        if inc.empty: return None
        
        inc = inc.iloc[-3:]
        col_map = {'Total Revenue': 'Receita', 'Operating Revenue': 'Receita', 'Cost Of Revenue': 'CPV', 
                   'Gross Profit': 'Lucro Bruto', 'Operating Income': 'EBIT', 'Net Income': 'Lucro Líquido'}
        
        df_final = pd.DataFrame(index=inc.index)
        rev_col = 'Total Revenue' if 'Total Revenue' in inc.columns else 'Operating Revenue'
        
        if rev_col in inc.columns:
            receita = inc[rev_col]
            for y_col, u_col in col_map.items():
                if y_col in inc.columns:
                    val = inc[y_col]
                    df_final[f'{u_col} (M)'] = val / 1e6
                    df_final[f'{u_col} AV%'] = (val / receita) * 100
                    df_final[f'{u_col} AH%'] = val.pct_change() * 100
            return df_final.T
        return None
    except: return None

# --- 4. INTERFACE ---
with st.spinner("Conectando ao Terminal..."):
    df_full = get_market_data()

if df_full.empty:
    st.error("Sem dados. Tente recarregar.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Filtros")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setor = st.selectbox("Setor", ["Todos"] + sorted(df_full['Setor'].unique().tolist()))
    
    with st.expander("📊 Indicadores", expanded=True):
        liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 1000000, 5000000, 10000000], value=500000)
        pl_r = st.slider("P/L", -10.0, 50.0, (-5.0, 30.0))
        dy_r = st.slider("DY %", 0.0, 20.0, (0.0, 20.0))
        roe_m = st.slider("ROE Min", -20.0, 40.0, 0.0)
    
    usar_yahoo = st.checkbox("Carregar Dados Contábeis (Yahoo)", value=True)

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

# Layout Principal
st.subheader(f"📋 Screener ({len(df_view)} ativos)")

t_main = st.tabs(["Geral", "💰 Dividendos", "💎 Valor", "✨ Magic Formula", "🛡️ Qualidade", "🚀 Crescimento"])

cols_main = ['Ticker', 'Setor', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'Graham_Fair', 'Upside', 'Score_Magic', 'Quality_Score']

cfg = {
    "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
    "PL": st.column_config.NumberColumn("P/L", format="%.1f"),
    "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
    "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
    "MargemLiquida": st.column_config.NumberColumn("Margem", format="%.1f%%"),
    "Div_Patrimonio": st.column_config.NumberColumn("Dívida/PL", format="%.2f"),
    "Graham_Fair": st.column_config.NumberColumn("Justo", format="R$ %.2f"),
    "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%"),
    "Quality_Score": st.column_config.NumberColumn("Qualidade", format="%.0f"),
    "Cresc_5a": st.column_config.NumberColumn("CAGR 5a", format="%.1f%%")
}

sel_ticker = None

def render_table(df_in, key):
    cols_exist = [c for c in cols_main if c in df_in.columns]
    ev = st.dataframe(df_in[cols_exist], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=400, key=key)
    if len(ev.selection.rows) > 0: return df_in.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t_main[0]: sel_ticker = render_table(df_view.sort_values('Liquidez', ascending=False), 't1')
with t_main[1]: sel_ticker = render_table(df_view.nlargest(100, 'DY'), 't2')
with t_main[2]: sel_ticker = render_table(df_view[(df_view['Upside']>0)].nlargest(100, 'Upside'), 't3')
with t_main[3]: sel_ticker = render_table(df_view.nsmallest(100, 'Score_Magic'), 't4')
with t_main[4]: sel_ticker = render_table(df_view.nlargest(100, 'Quality_Score'), 't5')
with t_main[5]: sel_ticker = render_table(df_view.nlargest(100, 'Cresc_5a'), 't6')

st.divider()

if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    st.markdown(f"## 🔬 Análise: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Preço", f"R$ {row['Preco']:.2f}")
    c2.metric("P/L", f"{row['PL']:.1f}x")
    c3.metric("P/VP", f"{row['PVP']:.2f}x")
    c4.metric("ROE", f"{row['ROE']:.1f}%")
    c5.metric("Dívida/PL", f"{row['Div_Patrimonio']:.2f}")
    
    td1, td2, td3 = st.tabs(["📊 Matriz Contábil", "📈 Gráfico", "💎 Valuation"])
    
    with td1:
        if usar_yahoo:
            with st.spinner("Gerando Matriz..."):
                df_matrix = get_accounting_matrix(sel_ticker)
                if df_matrix is not None:
                    st.info("Valores em Milhões (M). AV% = Margem (Vertical). AH% = Crescimento (Horizontal).")
                    st.dataframe(df_matrix.style.format("{:,.2f}"), use_container_width=True, height=500)
                else: st.warning("Dados indisponíveis no Yahoo.")
        else: st.info("Ative Yahoo.")
        
    with td2:
        if usar_yahoo:
            try:
                h = yf.download(sel_ticker+".SA", period="3y", progress=False)
                if not h.empty:
                    if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                    fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
                    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
            except: pass

    with td3:
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            vals = pd.DataFrame({'Modelo': ['Atual', 'Graham', 'Bazin'], 'Valor': [row['Preco'], row['Graham_Fair'], row['Bazin_Fair']]})
            fig_v = px.bar(vals, x='Modelo', y='Valor', color='Modelo', title="Valuation", template="plotly_dark")
            st.plotly_chart(fig_v, use_container_width=True)
        with c_v2:
            st.markdown("#### Setor")
            df_s = df_view[df_view['Setor'] == row['Setor']]
            fig_s = px.scatter(df_s, x='PL', y='ROE', size='Liquidez', color='DY', hover_name='Ticker', title=f"Setor: {row['Setor']}", template="plotly_dark")
            fig_s.add_annotation(x=row['PL'], y=row['ROE'], text="ESTE", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_s, use_container_width=True)

else:
    st.info("👆 Selecione um ativo na tabela.")
