import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO DE TERMINAL ---
st.set_page_config(page_title="Titanium Pro IX", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    
    /* Métricas */
    [data-testid="stMetricValue"] { font-size: 1.3rem; color: #00ffbf; font-family: 'Roboto Mono', monospace; font-weight: 700; }
    
    /* Tabelas Densas */
    .stDataFrame { border: 1px solid #30363d; border-radius: 5px; }
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #161b22; padding: 5px; border-radius: 6px; }
    .stTabs [data-baseweb="tab"] { height: 35px; border: none; color: #8b949e; font-weight: 600; font-size: 13px; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro IX: Accounting Master")

# --- 2. MOTOR DE DADOS BLINDADO (FUNDAMENTUS) ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Mapeamento Massivo (Todos os indicadores úteis)
        rename_map = {
            'Papel': 'Ticker', 'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'PSR': 'PSR',
            'Div.Yield': 'DY', 'P/Ativo': 'P_Ativo', 'P/Cap.Giro': 'P_CapGiro',
            'P/EBIT': 'P_EBIT', 'P/Ativ Circ Liq': 'P_AtivCircLiq',
            'EV/EBIT': 'EV_EBIT', 'EV/EBITDA': 'EV_EBITDA', 'Mrg Ebit': 'MargemEbit',
            'Mrg. Líq.': 'MargemLiquida', 'Liq. Corr.': 'LiqCorrente',
            'ROIC': 'ROIC', 'ROE': 'ROE', 'Liq.2meses': 'Liquidez',
            'Patrim. Líq': 'Patrimonio', 'Dív.Brut/ Patr.': 'Div_Patrimonio',
            'Cresc. Rec.5a': 'Cresc_5a', 'Ativo': 'Ativos'
        }
        
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        # Limpeza Numérica
        for col in df.columns:
            if col != 'Ticker':
                if df[col].dtype == object: df[col] = df[col].apply(clean_float)
        
        # Ajuste de Escala Percentual
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # Setores
        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO','KEPL']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings Quantitativos
        # 1. Valuation
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0)
        
        # 2. Magic Formula
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999
            
        # 3. Quality Score (Proprietário)
        # ROE Alto + Margem Alta + Dívida Baixa
        df['Quality_Score'] = (df['ROE'].rank(ascending=True) + 
                               df['MargemLiquida'].rank(ascending=True) + 
                               (df['Div_Patrimonio'] * -1).rank(ascending=True)) # Dívida menor é melhor

        return df
    except Exception as e:
        st.error(f"Erro Dados: {e}")
        return pd.DataFrame()

# --- 3. MOTOR CONTÁBIL AVANÇADO (YAHOO) ---
def get_accounting_matrix(ticker):
    """Gera a matriz de Análise Vertical e Horizontal para 3 anos"""
    try:
        stock = yf.Ticker(ticker+".SA")
        inc = stock.financials.T.sort_index(ascending=True) # DRE Anual
        
        if inc.empty: return None
        
        # Filtra últimos 3 anos
        inc = inc.iloc[-3:]
        
        # Tenta mapear colunas (Yahoo muda nomes as vezes)
        col_map = {
            'Total Revenue': 'Receita',
            'Operating Revenue': 'Receita', # Fallback
            'Cost Of Revenue': 'CPV',
            'Gross Profit': 'Lucro Bruto',
            'Operating Expense': 'Despesas Oper.',
            'Operating Income': 'EBIT',
            'Net Income': 'Lucro Líquido'
        }
        
        # Cria DF limpo
        df_final = pd.DataFrame(index=inc.index)
        
        # Encontra a coluna de Receita primeiro (Base 100%)
        rev_col = 'Total Revenue' if 'Total Revenue' in inc.columns else 'Operating Revenue'
        if rev_col not in inc.columns: return None
        
        receita = inc[rev_col]
        
        # Monta a matriz
        for yahoo_col, user_col in col_map.items():
            if yahoo_col in inc.columns:
                val = inc[yahoo_col]
                
                # 1. Valor Absoluto (em Milhões)
                df_final[f'{user_col} (M)'] = val / 1_000_000
                
                # 2. Análise Vertical (%) - Quanto representa da receita?
                df_final[f'{user_col} AV%'] = (val / receita) * 100
                
                # 3. Análise Horizontal (%) - Quanto cresceu ante o ano anterior?
                df_final[f'{user_col} AH%'] = val.pct_change() * 100
                
        # Transpõe para ficar Anos nas Colunas (Estilo Balanço)
        return df_final.T
            
    except: return None

# --- 4. INTERFACE ---
with st.spinner("Inicializando Terminal Quantitativo..."):
    df_full = get_market_data()

if df_full.empty:
    st.error("Sem conexão.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Filtros Pro")
    busca = st.text_input("Ticker", placeholder="PETR4").upper()
    setores = ["Todos"] + sorted(df_full['Setor'].unique().tolist())
    setor = st.selectbox("Setor", setores)
    
    with st.expander("📊 Indicadores", expanded=True):
        liq_min = st.select_slider("Liquidez", options=[0, 100000, 500000, 1000000, 5000000, 10000000], value=500000)
        pl_r = st.slider("P/L", -10.0, 50.0, (-5.0, 30.0))
        dy_r = st.slider("DY %", 0.0, 20.0, (0.0, 20.0))
        roe_m = st.slider("ROE Min", -20.0, 40.0, 0.0)
    
    usar_yahoo = st.checkbox("Carregar Dados Contábeis (Yahoo)", value=True)

# FILTROS
mask = (
    (df_full['Liquidez'] >= liq_min) &
    (df_full['PL'].between(pl_r[0], pl_r[1])) &
    (df_full['DY'].between(dy_r[0], dy_r[1])) &
    (df_full['ROE'] >= roe_m)
)
df_view = df_full[mask].copy()

if setor != "Todos": df_view = df_view[df_view['Setor'] == setor]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# --- LAYOUT ---
st.subheader(f"📋 Screener ({len(df_view)} ativos)")

# Abas de Rankings
t_main = st.tabs(["Geral", "💰 Dividendos", "💎 Valor", "✨ Magic Formula", "🛡️ Qualidade", "🚀 Crescimento"])

# Configuração Colunas Tabela
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
    # Filtra colunas existentes
    safe_cols = [c for c in cols_main if c in df_in.columns]
    ev = st.dataframe(df_in[safe_cols], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=400, key=key)
    if len(ev.selection.rows) > 0: return df_in.iloc[ev.selection.rows[0]]['Ticker']
    return None

with t_main[0]: # Geral
    sel_ticker = render_table(df_view.sort_values('Liquidez', ascending=False), 't1')
with t_main[1]: # Dividendos
    sel_ticker = render_table(df_view.nlargest(100, 'DY'), 't2')
with t_main[2]: # Valor
    sel_ticker = render_table(df_view[(df_view['Upside']>0)&(df_view['Upside']<500)].nlargest(100, 'Upside'), 't3')
with t_main[3]: # Magic
    sel_ticker = render_table(df_view.nsmallest(100, 'Score_Magic'), 't4')
with t_main[4]: # Qualidade (Novo)
    sel_ticker = render_table(df_view.nlargest(100, 'Quality_Score'), 't5')
with t_main[5]: # Crescimento (Novo)
    sel_ticker = render_table(df_view.nlargest(100, 'Cresc_5a'), 't6')

# --- DETALHES ---
st.divider()

if sel_ticker:
    row = df_full[df_full['Ticker'] == sel_ticker].iloc[0]
    st.markdown(f"## 🔬 Análise: <span style='color:#00ffbf'>{sel_ticker}</span>", unsafe_allow_html=True)
    
    # KPIs Topo
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Preço", f"R$ {row['Preco']:.2f}")
    c2.metric("P/L", f"{row['PL']:.1f}x")
    c3.metric("P/VP", f"{row['PVP']:.2f}x")
    c4.metric("ROE", f"{row['ROE']:.1f}%")
    c5.metric("Dívida/PL", f"{row.get('Div_Patrimonio',0):.2f}")
    
    # Abas de Detalhe
    td1, td2, td3 = st.tabs(["📊 Matriz Contábil (AV/AH)", "📈 Gráfico & Técnica", "💎 Valuation"])
    
    with td1:
        if usar_yahoo:
            with st.spinner("Gerando Matriz Contábil..."):
                df_matrix = get_accounting_matrix(sel_ticker)
                
                if df_matrix is not None:
                    st.info("Valores em Milhões (M). AV% = Análise Vertical (% da Receita). AH% = Análise Horizontal (Crescimento Ano a Ano).")
                    # Formatação condicional
                    st.dataframe(df_matrix.style.format("{:,.2f}"), use_container_width=True, height=500)
                else:
                    st.warning("Dados contábeis detalhados não disponíveis no Yahoo Finance para este ativo.")
        else: st.info("Ative a opção Yahoo na barra lateral.")
        
    with td2:
        if usar_yahoo:
            try:
                hist = yf.download(sel_ticker+".SA", period="3y", progress=False)
                if not hist.empty:
                    if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.droplevel(1)
                    
                    fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
                    fig.update_layout(title="Price Action (3 Anos)", template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
            except: st.error("Gráfico indisponível.")
            
    with td3:
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            vals = pd.DataFrame({'Modelo': ['Atual', 'Graham (Justo)', 'Bazin (Teto)'], 'Valor': [row['Preco'], row['Graham_Fair'], row['Bazin_Fair']]})
            fig_v = px.bar(vals, x='Modelo', y='Valor', color='Modelo', title="Valuation", template="plotly_dark")
            st.plotly_chart(fig_v, use_container_width=True)
        with c_v2:
            st.markdown("#### Indicadores Extras")
            st.write(f"**Crescimento 5 Anos (CAGR):** {row.get('Cresc_5a',0):.2f}%")
            st.write(f"**Liquidez Corrente:** {row.get('LiqCorrente',0):.2f}")
            st.write(f"**Margem EBIT:** {row.get('MargemEbit',0):.2f}%")

else:
    st.info("👆 Selecione um ativo na tabela para ver a Matriz Contábil.")
