import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

# --- 1. CONFIGURAÇÃO VISUAL DARK/NEON ---
st.set_page_config(page_title="Titanium X | Terminal", layout="wide", initial_sidebar_state="collapsed")

# CSS para visual de Terminal Financeiro
st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    /* Métricas com cor neon */
    [data-testid="stMetricValue"] { font-size: 1.3rem; color: #00f2ea; font-family: 'Roboto Mono', monospace; }
    [data-testid="stMetricLabel"] { color: #888; }
    
    /* Abas estilo profissional */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #161b22; padding: 5px; border-radius: 5px; }
    .stTabs [data-baseweb="tab"] {
        height: 35px; background-color: transparent; color: #aaa; border: none; font-size: 13px;
    }
    .stTabs [aria-selected="true"] { background-color: #262d3d !important; color: #00f2ea !important; border-bottom: 2px solid #00f2ea; }
    
    /* Tabelas */
    [data-testid="stDataFrame"] { border: 1px solid #333; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #111418; border-right: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium X: Hedge Fund Terminal")

# --- 2. FUNÇÕES MATEMÁTICAS & DADOS ---

def calcular_rsi(series, period=14):
    """Calcula o Índice de Força Relativa (RSI) manualmente"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def limpar_coluna(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def carregar_dados_base():
    try:
        # Base Fundamentus
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        mapa = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a',
            'Patrim. Líq': 'Patrimonio', 'Ativo': 'Ativos'
        }
        
        cols = ['Ticker'] + [c for c in mapa.keys() if c in df.columns]
        df = df[cols].copy()
        df.rename(columns=mapa, inplace=True)
        
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(limpar_coluna)
            
        # Ajustes Percentuais
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        # Classificação de Setor (Manual simplificada para velocidade)
        def get_setor(t):
            t = t.upper()
            if t.startswith(('ITUB','BBDC','BBAS','SANB','BPAC')): return 'Finanças'
            if t.startswith(('VALE','CSNA','GGBR','USIM')): return 'Materiais Básicos'
            if t.startswith(('PETR','PRIO','UGPA','CSAN')): return 'Energia'
            if t.startswith(('MGLU','LREN','ARZZ','PETZ')): return 'Consumo Cíclico'
            if t.startswith(('WEGE','EMBR','TUPY')): return 'Industrial'
            if t.startswith(('TAEE','TRPL','ELET','CPLE')): return 'Utilidade Pública'
            if t.startswith(('RADL','RDOR','HAPV')): return 'Saúde'
            return 'Outros'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Cálculos de Valuation Automáticos
        df['Graham'] = np.where((df['PL']>0)&(df['PVP']>0), np.sqrt(22.5 * (df['Preco']/df['PL']) * (df['Preco']/df['PVP'])), 0)
        df['Upside'] = np.where((df['Graham']>0), ((df['Graham']-df['Preco'])/df['Preco'])*100, -999)
        df['Bazin'] = np.where(df['DY']>0, df['Preco']*(df['DY']/6), 0)
        
        # Magic Formula Score
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Rank_EV'] = df_m['EV_EBIT'].rank(ascending=True)
            df_m['Rank_ROIC'] = df_m['ROIC'].rank(ascending=False)
            df_m['Score_Magic'] = df_m['Rank_EV'] + df_m['Rank_ROIC']
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
            df['Score_Magic'] = df['Score_Magic'].fillna(99999)
        else:
            df['Score_Magic'] = 99999
            
        return df
    except: return pd.DataFrame()

# --- 3. INTERFACE LATERAL ---
st.sidebar.header("🎛️ Centro de Controle")
usar_tech = st.sidebar.checkbox("📡 Dados Técnicos (Yahoo)", value=True)

with st.spinner('Carregando Core...'):
    df = carregar_dados_base()

if df.empty:
    st.error("Erro ao conectar com a B3.")
    st.stop()

# Filtros Globais
busca = st.sidebar.text_input("Ticker", placeholder="PETR4").upper()
setor_f = st.sidebar.selectbox("Setor", ["Todos"] + sorted(df['Setor'].unique().tolist()))
liq_f = st.sidebar.select_slider("Liquidez Mínima", options=[0, 100000, 1000000, 5000000], value=200000)

# Aplica Filtros
df_view = df[df['Liquidez'] >= liq_f].copy()
if setor_f != "Todos": df_view = df_view[df_view['Setor'] == setor_f]
if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

# Cálculo de Momentum Lote (Para tabela)
if usar_tech:
    with st.spinner("Calculando Momentum..."):
        # Pega Top 50 para não travar
        top_m = df_view.nlargest(50, 'Liquidez')['Ticker'].tolist()
        tickers_sa = [t+".SA" for t in top_m]
        try:
            data = yf.download(tickers_sa, period="6mo", progress=False)['Adj Close']
            # Tratamento para 1 ticker vs Múltiplos
            if isinstance(data, pd.Series): data = data.to_frame(name=tickers_sa[0])
            
            res = {}
            for t in data.columns:
                s = data[t].dropna()
                if len(s) > 10:
                    ret = ((s.iloc[-1] - s.iloc[0]) / s.iloc[0]) * 100
                    res[t.replace('.SA','')] = ret
            
            df_view['Momentum_6M'] = df_view['Ticker'].map(res).fillna(0)
        except:
            df_view['Momentum_6M'] = 0

# --- 4. LAYOUT PRINCIPAL ---

# KPI Bar
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ativos", len(df_view))
k2.metric("Yield Médio", f"{df_view[df_view['DY']>0]['DY'].mean():.2f}%")
k3.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
k4.metric("ROE Médio", f"{df_view['ROE'].mean():.1f}%")
try:
    best_sec = df_view.groupby('Setor')['DY'].mean().idxmax()
    k5.metric("Melhor Setor (DY)", best_sec)
except: k5.metric("Setor", "-")

st.divider()

col_tabela, col_dash = st.columns([1.5, 2.5])

# --- COLUNA ESQUERDA: SELETOR DE AÇÕES ---
with col_tabela:
    st.subheader("📋 Screener de Ativos")
    
    tab_a, tab_b, tab_c = st.tabs(["Geral", "Dividendos", "Valor"])
    
    cfg = {
        "Preco": st.column_config.NumberColumn("R$", format="%.2f"),
        "DY": st.column_config.ProgressColumn("DY", format="%.1f%%", min_value=0, max_value=15),
        "Momentum_6M": st.column_config.NumberColumn("Mom.", format="%.1f%%"),
        "Upside": st.column_config.NumberColumn("Upside", format="%.0f%%")
    }
    
    sel_row = None
    
    with tab_a:
        df_sort = df_view.sort_values('Liquidez', ascending=False).head(100)
        ev = st.dataframe(df_sort[['Ticker','Preco','Momentum_6M','DY']], column_config=cfg, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if len(ev.selection.rows)>0: sel_row = df_sort.iloc[ev.selection.rows[0]]
        
    with tab_b:
        df_sort = df_view.nlargest(100, 'DY')
        ev = st.dataframe(df_sort[['Ticker','Preco','DY','Bazin']], column_config={**cfg, "Bazin": st.column_config.NumberColumn("Teto", format="%.2f")}, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if len(ev.selection.rows)>0: sel_row = df_sort.iloc[ev.selection.rows[0]]

    with tab_c:
        df_sort = df_view[(df_view['Upside']>0)&(df_view['Upside']<500)].nlargest(100, 'Upside')
        ev = st.dataframe(df_sort[['Ticker','Preco','Graham','Upside']], column_config={**cfg, "Graham": st.column_config.NumberColumn("Justo", format="%.2f")}, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if len(ev.selection.rows)>0: sel_row = df_sort.iloc[ev.selection.rows[0]]

# --- COLUNA DIREITA: 10 GRÁFICOS PROFISSIONAIS ---
with col_dash:
    if sel_row is not None:
        # DADOS DO ATIVO SELECIONADO
        ticker = sel_row['Ticker']
        st.markdown(f"## 🔎 Análise Profunda: <span style='color:#00f2ea'>{ticker}</span>", unsafe_allow_html=True)
        
        # Baixar Histórico Longo (10 ANOS)
        with st.spinner(f"Baixando histórico de 10 anos de {ticker}..."):
            try:
                stock = yf.Ticker(ticker+".SA")
                hist = stock.history(period="10y")
                
                if hist.empty:
                    st.error("Dados históricos não encontrados.")
                else:
                    # Cálculo de Indicadores Técnicos
                    hist['RSI'] = calcular_rsi(hist['Close'])
                    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                    
                    # --- ABA DE GRÁFICOS ---
                    g1, g2, g3 = st.tabs(["📈 Técnico & Preço", "📊 Fundamentalista", "🧠 Comparativo"])
                    
                    with g1:
                        # GRÁFICO 1: CANDLESTICK COMPLETO COM MÉDIAS
                        fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                        
                        # Candles
                        fig_candle.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Preço'), row=1, col=1)
                        # Médias Móveis
                        fig_candle.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='orange', width=1), name='Média 50'), row=1, col=1)
                        fig_candle.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='cyan', width=1), name='Média 200'), row=1, col=1)
                        
                        # RSI (GRÁFICO 2 EMBUTIDO)
                        fig_candle.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple', width=1), name='RSI'), row=2, col=1)
                        # Linhas de Sobrecompra/Sobrevenda
                        fig_candle.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig_candle.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                        
                        fig_candle.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, title="Price Action + RSI (10 Anos)")
                        st.plotly_chart(fig_candle, use_container_width=True)
                        
                        # GRÁFICO 3: VOLUME POR ANO
                        hist['Year'] = hist.index.year
                        vol_year = hist.groupby('Year')['Volume'].sum().reset_index()
                        fig_vol = px.bar(vol_year, x='Year', y='Volume', title="Volume Negociado por Ano", template="plotly_dark")
                        st.plotly_chart(fig_vol, use_container_width=True)

                    with g2:
                        c_f1, c_f2 = st.columns(2)
                        
                        # GRÁFICO 4: MARGENS (Barra)
                        margins = pd.DataFrame({
                            'Tipo': ['Margem Bruta', 'Margem EBIT', 'Margem Líquida'],
                            'Valor': [sel_row.get('MargemEbit', 0)*1.2, sel_row.get('MargemEbit', 0), sel_row.get('MargemLiquida', 0)] # Estimando bruta
                        })
                        fig_marg = px.bar(margins, x='Tipo', y='Valor', color='Valor', title="Estrutura de Margens", template="plotly_dark", color_continuous_scale='Viridis')
                        c_f1.plotly_chart(fig_marg, use_container_width=True)
                        
                        # GRÁFICO 5: ENDIVIDAMENTO (Gauge)
                        div_patr = sel_row.get('Div_Patrimonio', 0)
                        fig_div = go.Figure(go.Indicator(
                            mode = "gauge+number", value = div_patr,
                            title = {'text': "Dívida / Patrimônio"},
                            gauge = {'axis': {'range': [None, 5]}, 
                                     'bar': {'color': "red" if div_patr > 3 else "green"},
                                     'steps': [{'range': [0, 1], 'color': "gray"}, {'range': [1, 3], 'color': "lightgray"}]}
                        ))
                        fig_div.update_layout(height=300, margin=dict(t=50,b=0,l=20,r=20), template="plotly_dark")
                        c_f2.plotly_chart(fig_div, use_container_width=True)
                        
                        # GRÁFICO 6: COMPARATIVO VALUATION
                        metrics = pd.DataFrame({
                            'Métrica': ['P/L', 'P/VP', 'EV/EBIT'],
                            'Valor': [sel_row['PL'], sel_row['PVP'], sel_row['EV_EBIT']]
                        })
                        fig_val = px.bar(metrics, x='Métrica', y='Valor', title="Múltiplos de Valuation", template="plotly_dark")
                        st.plotly_chart(fig_val, use_container_width=True)

                    with g3:
                        # GRÁFICO 7: RISCO X RETORNO (SCATTER DO SETOR)
                        df_setor = df_view[df_view['Setor'] == sel_row['Setor']]
                        fig_scat = px.scatter(df_setor, x='PL', y='ROE', size='Liquidez', color='Ticker', 
                                              hover_name='Ticker', title=f"Comparativo Setor: {sel_row['Setor']}", template="plotly_dark")
                        # Destaca a ação selecionada
                        fig_scat.add_annotation(x=sel_row['PL'], y=sel_row['ROE'], text="ESTA AÇÃO", showarrow=True, arrowhead=1)
                        st.plotly_chart(fig_scat, use_container_width=True)
                        
                        # GRÁFICO 8: HISTOGRAMA DE P/L
                        fig_hist = px.histogram(df_view, x="PL", nbins=40, title="Distribuição de P/L do Mercado", template="plotly_dark")
                        fig_hist.add_vline(x=sel_row['PL'], line_color="red", annotation_text="Ativo")
                        st.plotly_chart(fig_hist, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao gerar gráficos: {e}")
                
    else:
        # TELA DE STANDBY (QUANDO NADA SELECIONADO)
        st.info("👈 Selecione um ativo na tabela para abrir o Data Center.")
        
        # GRÁFICO 9: TREEMAP DE SETORES
        st.subheader("Visão Macro: Setores")
        df_tree = df_view.groupby('Setor')[['Liquidez', 'DY']].mean().reset_index()
        df_tree['Count'] = df_view.groupby('Setor')['Ticker'].count().values
        fig_tree = px.treemap(df_tree, path=['Setor'], values='Count', color='DY', 
                              title="Setores por Qtde de Empresas (Cor = Yield Médio)", template="plotly_dark")
        st.plotly_chart(fig_tree, use_container_width=True)
        
        # GRÁFICO 10: TOP YIELDS
        st.subheader("Top Yields do Mercado")
        top_y = df_view.nlargest(15, 'DY')
        fig_bar = px.bar(top_y, x='Ticker', y='DY', color='Setor', title="Maiores Pagadores", template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)
