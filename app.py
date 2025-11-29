import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Titanium Pro V", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 1.2rem; color: #00e676; }
    div.stDataFrame div[data-testid="stDataFrame"] { border: 1px solid #333; }
    /* Ajuste para seleção na tabela ficar mais visível */
    .stDataFrame { border: 1px solid #00e676; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro V: Robust Edition")

# --- 2. MOTOR DE DADOS BLINDADO (FUNDAMENTUS) ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data_stealth():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # Renomeação
        rename_map = {
            'Papel': 'Ticker', 'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 
            'Div.Yield': 'DY', 'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida', 
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        # Filtra e renomeia
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        # Limpeza Numérica
        for col in df.columns:
            if col != 'Ticker':
                if df[col].dtype == object:
                    df[col] = df[col].apply(clean_float)
        
        # Ajustes Percentuais (Fundamentus às vezes vem 10,0 que vira 10.0, ok. 
        # Se vier 0.10, multiplicamos. Vamos assumir padrão visual percentual)
        
        # Classificação Setorial (Simplificada para performance)
        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL']: return 'Elétricas'
            return 'Geral'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings
        df['Graham'] = np.where((df['PL']>0)&(df['PVP']>0), np.sqrt(22.5 * (df['Preco']/df['PL']) * (df['Preco']/df['PVP'])), 0)
        df['Upside'] = np.where((df['Graham']>0), ((df['Graham']-df['Preco'])/df['Preco'])*100, -999)
        
        return df
    except Exception as e:
        st.error(f"Erro Fundamentus: {e}")
        return pd.DataFrame()

# --- 3. MOTOR DE MOMENTUM (YAHOO CORRIGIDO) ---
@st.cache_data(ttl=1800, show_spinner=False)
def calculate_momentum_batch(tickers_list):
    """Baixa dados do Yahoo e calcula Momentum de 6 meses"""
    if not tickers_list: return {}
    
    tickers_sa = [t + ".SA" for t in tickers_list]
    momentum_dict = {}
    
    try:
        # Baixa 7 meses para garantir 6 meses de dados
        data = yf.download(tickers_sa, period="7mo", progress=False)['Adj Close']
        
        # Se for apenas 1 ticker, o Yahoo retorna Series, transformamos em DF
        if isinstance(data, pd.Series): 
            data = data.to_frame(name=tickers_sa[0])
        
        for col in data.columns:
            try:
                series = data[col].dropna()
                if len(series) > 20:
                    start_price = series.iloc[0]
                    end_price = series.iloc[-1]
                    mom = ((end_price - start_price) / start_price) * 100
                    
                    clean_ticker = col.replace('.SA', '')
                    momentum_dict[clean_ticker] = mom
            except: pass
            
    except Exception as e:
        print(f"Erro Yahoo: {e}")
        
    return momentum_dict

# --- 4. INTERFACE ---
st.sidebar.header("🎛️ Centro de Comando")
usar_yahoo = st.sidebar.checkbox("📡 Ativar Momentum (Yahoo)", value=True)

with st.spinner('Conectando ao Mercado...'):
    df_full = get_market_data_stealth()

if df_full.empty:
    st.error("Sem dados. Tente recarregar.")
    st.stop()

# Filtros
busca = st.sidebar.text_input("Ticker", placeholder="PETR4").upper()
setor_f = st.sidebar.selectbox("Setor", ["Todos"] + sorted(df_full['Setor'].unique().tolist()))
liq_f = st.sidebar.select_slider("Liquidez Mínima", options=[0, 100000, 200000, 1000000, 5000000], value=200000)

# Aplicação Filtros
df_view = df_full[df_full['Liquidez'] >= liq_f].copy()
if setor_f != "Todos": df_view = df_view[df_view['Setor'] == setor_f]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Lógica de Momentum
if usar_yahoo:
    with st.spinner("Calculando Momentum (Top 50 Liquidez)..."):
        # Limita aos top 50 filtrados para performance
        top_tickers = df_view.nlargest(50, 'Liquidez')['Ticker'].tolist()
        mom_data = calculate_momentum_batch(top_tickers)
        # Mapeia. Quem não foi calculado (por não ser top 50 ou erro) fica com NaN -> 0
        df_view['Momentum'] = df_view['Ticker'].map(mom_data).fillna(0)
else:
    df_view['Momentum'] = 0

# --- LAYOUT PRINCIPAL ---
col_L, col_R = st.columns([1.5, 2.5])

# Variável para guardar a linha completa selecionada
selected_full_row = None

with col_L:
    st.subheader("📋 Screener")
    t1, t2, t3, t4 = st.tabs(["Geral", "🚀 Momentum", "💰 Dividendos", "💎 Valor"])
    
    cfg = {
        "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
        "DY": st.column_config.ProgressColumn("DY", format="%.1f%%", min_value=0, max_value=15),
        "Momentum": st.column_config.NumberColumn("Mom. 6m", format="%.1f%%"),
        "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%")
    }
    
    # Função auxiliar para renderizar e capturar seleção
    def render_tab_table(df_input, cols_show, key_id):
        event = st.dataframe(
            df_input[cols_show],
            column_config=cfg,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=500,
            key=key_id
        )
        if len(event.selection.rows) > 0:
            # Pega o índice da linha selecionada na tabela VISUAL
            idx_visual = event.selection.rows[0]
            # Pega o Ticker correspondente nessa linha
            ticker_sel = df_input.iloc[idx_visual]['Ticker']
            # CORREÇÃO DO KEYERROR: Busca a linha completa no DF Original usando o Ticker
            return df_full[df_full['Ticker'] == ticker_sel].iloc[0]
        return None

    with t1: # Geral
        d = df_view.sort_values('Liquidez', ascending=False).head(100)
        s = render_tab_table(d, ['Ticker', 'Preco', 'Momentum', 'DY'], 't1')
        if s is not None: selected_full_row = s
        
    with t2: # Momentum (Novo Ranking)
        if usar_yahoo:
            d = df_view.sort_values('Momentum', ascending=False).head(50)
            s = render_tab_table(d, ['Ticker', 'Preco', 'Momentum', 'PL'], 't2')
            if s is not None: selected_full_row = s
        else: st.warning("Ative Yahoo para ver Momentum.")

    with t3: # Dividendos
        d = df_view.nlargest(100, 'DY')
        s = render_tab_table(d, ['Ticker', 'Preco', 'DY', 'PVP'], 't3')
        if s is not None: selected_full_row = s

    with t4: # Valor
        d = df_view[(df_view['Upside']>0)&(df_view['Upside']<500)].nlargest(100, 'Upside')
        s = render_tab_table(d, ['Ticker', 'Preco', 'Graham', 'Upside'], 't4')
        if s is not None: selected_full_row = s

# --- PAINEL DE DETALHES ---
with col_R:
    if selected_full_row is not None:
        # Agora temos a linha COMPLETA (com PL, PVP, etc), não apenas a visual
        row = selected_full_row 
        tk = row['Ticker']
        
        st.markdown(f"## 🔎 Análise: <span style='color:#00f2ea'>{tk}</span>", unsafe_allow_html=True)
        
        # Métricas Principais (Agora seguro contra KeyError)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço", f"R$ {row['Preco']:.2f}")
        c2.metric("P/L", f"{row['PL']:.1f}x")
        c3.metric("P/VP", f"{row['PVP']:.2f}x")
        
        # Momentum só aparece se foi calculado, senão mostra traço
        mom_val = df_view.loc[df_view['Ticker']==tk, 'Momentum'].values[0] if 'Momentum' in df_view.columns else 0
        c4.metric("Momentum", f"{mom_val:.1f}%")
        
        st.divider()
        
        # Gráficos
        tab_g, tab_f = st.tabs(["📈 Gráfico Técnico", "📊 Fundamentos"])
        
        with tab_g:
            if usar_yahoo:
                with st.spinner("Baixando Histórico..."):
                    try:
                        h = yf.download(tk+".SA", period="2y", progress=False)
                        if not h.empty:
                            if isinstance(h.columns, pd.MultiIndex): h.columns = h.columns.droplevel(1)
                            
                            fig = go.Figure(data=[go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'])])
                            fig.update_layout(title="Price Action (2 Anos)", template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.warning("Dados indisponíveis no Yahoo.")
                    except: st.error("Erro no gráfico.")
            else: st.info("Ative Yahoo para ver gráficos.")
            
        with tab_f:
            c_f1, c_f2 = st.columns(2)
            # Margens
            m_data = pd.DataFrame({
                'Tipo': ['Bruta (Est)', 'EBIT', 'Líquida'],
                'Valor': [row.get('MargemEbit',0)*1.3, row.get('MargemEbit',0), row.get('MargemLiquida',0)]
            })
            fig_m = px.bar(m_data, x='Tipo', y='Valor', color='Tipo', title="Margens (%)", template="plotly_dark")
            c_f1.plotly_chart(fig_m, use_container_width=True)
            
            # Dívida
            div = row.get('Div_Patrimonio', 0)
            fig_d = go.Figure(go.Indicator(
                mode="gauge+number", value=div, title={'text': "Dívida Líq / Patrimônio"},
                gauge={'axis': {'range': [None, 5]}, 'bar': {'color': "red" if div>3 else "green"}}
            ))
            fig_d.update_layout(height=300, margin=dict(t=40,b=0,l=20,r=20), template="plotly_dark")
            c_f2.plotly_chart(fig_d, use_container_width=True)

    else:
        st.info("👈 Selecione um ativo na tabela para ver o Raio-X completo.")
        
        # Dashboard Macro (Quando nada selecionado)
        st.subheader("Mapa Macro")
        try:
            tree_df = df_view.groupby('Setor')[['Liquidez', 'DY']].mean().reset_index()
            tree_df['Qtd'] = df_view.groupby('Setor')['Ticker'].count().values
            fig = px.treemap(tree_df, path=['Setor'], values='Qtd', color='DY', 
                             title="Setores (Tamanho=Qtd, Cor=Yield)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        except: pass
