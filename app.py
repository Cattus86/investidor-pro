import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Investidor Pro | Safe", layout="wide", initial_sidebar_state="expanded")

# CSS para garantir que a interface carregue bonito
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 1.2rem; color: #00e676; }
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background-color: #1f2937; color: #aaa; border-radius: 4px;
    }
    .stTabs [aria-selected="true"] { background-color: #00e676 !important; color: black !important; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Investidor Pro: Ultimate (Safe Mode)")

# --- 2. FUNÇÕES DE SUPORTE ---
MAPA_SETORES = {
    'Bancos': ['BBAS3', 'ITUB4', 'BBDC4', 'SANB11', 'BPAC11', 'ABCB4', 'BRSR6', 'ITSA4'],
    'Energia': ['PETR4', 'PETR3', 'PRIO3', 'VBBR3', 'UGPA3', 'CSAN3', 'ENAT3', 'RRRP3', 'RECV3'],
    'Elétricas': ['ELET3', 'ELET6', 'EGIE3', 'TRPL4', 'TAEE11', 'CPLE6', 'CMIG4', 'EQTL3', 'NEOE3'],
    'Mineração/Sid': ['VALE3', 'CSNA3', 'GGBR4', 'GOAU4', 'USIM5', 'CMIN3', 'FESA4'],
    'Varejo': ['MGLU3', 'LREN3', 'ARZZ3', 'SOMA3', 'PETZ3', 'RDOR3', 'RADL3', 'AMER3'],
    'Outros': []
}

def obter_setor(ticker):
    t = ticker.upper().strip()
    for s, l in MAPA_SETORES.items():
        if t in l: return s
    return "Geral"

def limpar_coluna(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

# --- 3. MOTORES DE DADOS (SEPARADOS PARA NÃO TRAVAR) ---

@st.cache_data(ttl=600)
def carregar_fundamentalista():
    """Carrega APENAS dados leves do Fundamentus (Rápido)"""
    try:
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        mapa = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        cols = ['Ticker'] + [c for c in mapa.keys() if c in df.columns]
        df = df[cols].copy()
        df.rename(columns=mapa, inplace=True)
        
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(limpar_coluna)
            
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        df['Setor'] = df['Ticker'].apply(obter_setor)
        
        # Inicializa colunas técnicas com 0 (serão preenchidas depois se o usuário quiser)
        df['Momentum'] = 0.0
        df['Volatilidade'] = 0.0
        
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def carregar_tecnico_limitado(df_base, top_n=40):
    """
    Carrega Yahoo Finance APENAS para os Top N ativos mais líquidos.
    Isso evita o crash do servidor.
    """
    df_top = df_base.nlargest(top_n, 'Liquidez').copy()
    tickers = [t + ".SA" for t in df_top['Ticker'].tolist()]
    
    if not tickers: return df_base
    
    try:
        # Baixa apenas 6 meses para ser rápido
        dados = yf.download(tickers, period="6mo", progress=False)['Adj Close']
        
        if isinstance(dados, pd.Series): dados = dados.to_frame()
        
        res = {}
        for t_full in dados.columns:
            t_clean = t_full.replace('.SA', '')
            serie = dados[t_full].dropna()
            
            if len(serie) > 10:
                try:
                    # Momentum
                    p_curr = serie.iloc[-1]
                    p_prev = serie.iloc[0]
                    mom = ((p_curr - p_prev) / p_prev) * 100
                    
                    # Volatilidade (Simplificada)
                    ret = serie.pct_change().dropna()
                    vol = ret.std() * np.sqrt(252) * 100
                    
                    res[t_clean] = {'Momentum': mom, 'Volatilidade': vol}
                except: pass
        
        # Atualiza o DF original apenas onde temos dados
        for i, row in df_base.iterrows():
            if row['Ticker'] in res:
                df_base.at[i, 'Momentum'] = res[row['Ticker']]['Momentum']
                df_base.at[i, 'Volatilidade'] = res[row['Ticker']]['Volatilidade']
                
    except: pass
    
    return df_base

# --- 4. CÁLCULOS FINAIS ---
def calcular_rankings(df):
    # Graham
    df['LPA'] = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
    df['VPA'] = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
    mask = (df['LPA']>0) & (df['VPA']>0)
    df.loc[mask, 'Graham'] = np.sqrt(22.5 * df.loc[mask, 'LPA'] * df.loc[mask, 'VPA'])
    df['Graham'] = df['Graham'].fillna(0)
    df['Upside_Graham'] = np.where((df['Graham']>0) & (df['Preco']>0), ((df['Graham']-df['Preco'])/df['Preco'])*100, -999)
    
    # Magic Formula
    df_m = df[(df['EV_EBIT']>0) & (df['ROIC']>0)].copy()
    if not df_m.empty:
        df_m['Rank_EV'] = df_m['EV_EBIT'].rank(ascending=True)
        df_m['Rank_ROIC'] = df_m['ROIC'].rank(ascending=False)
        df_m['Score_Magic'] = df_m['Rank_EV'] + df_m['Rank_ROIC']
        df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
    else:
        df['Score_Magic'] = 99999
    
    # Bazin
    df['Bazin'] = np.where(df['DY']>0, df['Preco']*(df['DY']/6), 0)
    
    return df

def analisar_ia(row):
    score = 5
    txt = []
    # Lógica simplificada
    if row['PL'] < 5 and row['PL'] > 0: txt.append("🟢 P/L Baixo"); score+=2
    if row['DY'] > 8: txt.append("🟢 Yield Alto"); score+=1
    if row['ROE'] > 15: txt.append("🔥 ROE Alto"); score+=2
    if row['Momentum'] > 15: txt.append("🚀 Tendência Alta"); score+=1
    return score, " | ".join(txt)

# --- 5. EXECUÇÃO DO APP ---

# A. Carregamento Seguro (Sem Yahoo no start)
with st.spinner('Carregando base fundamentalista...'):
    df_raw = carregar_fundamentalista()

if not df_raw.empty:
    
    # B. Sidebar e Botão Turbo
    st.sidebar.header("⚙️ Configurações")
    
    # O Pulo do Gato: Carregar dados pesados apenas se o usuário deixar
    usar_yahoo = st.sidebar.checkbox("📡 Ativar Dados Técnicos (Yahoo)", value=True, help="Baixa dados de Momentum/Volatilidade para as Top 40 ações. Pode demorar uns segundos.")
    
    df_work = df_raw.copy()
    
    if usar_yahoo:
        with st.spinner('Baixando dados técnicos (Top 40)...'):
            df_work = carregar_tecnico_limitado(df_work, top_n=40)
            
    df = calcular_rankings(df_work)
    
    # Filtros
    busca = st.sidebar.text_input("Ticker:", placeholder="Ex: VALE3").upper().strip()
    f_setor = st.sidebar.selectbox("Setor:", ["Todos"] + sorted(list(df['Setor'].unique())))
    f_liq = st.sidebar.select_slider("Liquidez:", options=[0, 100000, 1000000], value=100000)
    
    mask = (df['Liquidez'] >= f_liq)
    df_view = df[mask].copy()
    if f_setor != "Todos": df_view = df_view[df_view['Setor'] == f_setor]
    if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]

    # C. Dashboard
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ativos", len(df_view))
    c2.metric("Yield Médio", f"{df_view[df_view['DY']>0]['DY'].mean():.2f}%")
    c3.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
    
    # Se tiver momentum, mostra destaque
    if usar_yahoo:
        try:
            hot = df_view.nlargest(1, 'Momentum').iloc[0]
            c4.metric(f"Top Trend: {hot['Ticker']}", f"+{hot['Momentum']:.1f}%")
        except: c4.metric("Top Trend", "-")
    else:
        c4.metric("Dados Técnicos", "OFF")

    st.divider()

    # D. Layout Principal
    col_main, col_det = st.columns([2, 1])
    
    if 'sel' not in st.session_state: st.session_state['sel'] = None
    def on_sel(evt, df_r):
        if len(evt.selection.rows)>0: st.session_state['sel'] = df_r.iloc[evt.selection.rows[0]]

    cfg = {
        "Preco": st.column_config.NumberColumn("R$", format="R$ %.2f"),
        "Momentum": st.column_config.ProgressColumn("Trend", format="%.1f%%", min_value=-30, max_value=30),
        "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
        "Graham": st.column_config.NumberColumn("Justo", format="R$ %.2f"),
        "Score_Magic": st.column_config.NumberColumn("Score", format="%d")
    }

    with col_main:
        tabs = st.tabs(["🚀 Momentum", "💰 Dividendos", "💎 Valor", "✨ Magic", "📊 Todos"])
        
        with tabs[0]:
            if usar_yahoo:
                df_t = df_view.sort_values('Momentum', ascending=False).head(50)
                ev = st.dataframe(df_t[['Ticker','Preco','Momentum','Volatilidade']], column_config=cfg, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
                on_sel(ev, df_t)
            else:
                st.warning("Ative 'Dados Técnicos' na barra lateral para ver este ranking.")
        
        with tabs[1]:
            df_t = df_view.nlargest(50, 'DY')
            ev = st.dataframe(df_t[['Ticker','Preco','DY','Bazin']], column_config={**cfg, "Bazin": st.column_config.NumberColumn("Teto", format="R$ %.2f")}, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
            on_sel(ev, df_t)
            
        with tabs[2]:
            df_t = df_view[(df_view['Upside_Graham']>0) & (df_view['Upside_Graham']<500)].nlargest(50, 'Upside_Graham')
            ev = st.dataframe(df_t[['Ticker','Preco','Graham','Upside_Graham','PL']], column_config={**cfg, "Upside_Graham": st.column_config.NumberColumn("Upside", format="%.0f%%")}, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
            on_sel(ev, df_t)

        with tabs[3]:
            df_t = df_view.nsmallest(50, 'Score_Magic')
            ev = st.dataframe(df_t[['Ticker','Preco','EV_EBIT','ROIC','Score_Magic']], column_config=cfg, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row")
            on_sel(ev, df_t)
            
        with tabs[4]:
            st.dataframe(df_view, column_config=cfg, hide_index=True, use_container_width=True)

    with col_det:
        st.markdown("### 🔬 Raio-X")
        if st.session_state['sel'] is not None:
            row = st.session_state['sel']
            score, txt = analisar_ia(row)
            
            st.markdown(f"# {row['Ticker']}")
            st.metric("Preço", f"R$ {row['Preco']:.2f}")
            
            # Gráfico ON DEMAND (Só baixa se clicar)
            if usar_yahoo:
                try:
                    with st.spinner('Gráfico...'):
                        h = yf.download(row['Ticker']+".SA", period="2y", progress=False)
                        if not h.empty:
                            # Ajuste para nova versão do yfinance que retorna MultiIndex
                            if isinstance(h.columns, pd.MultiIndex):
                                h.columns = h.columns.droplevel(1)
                                
                            fig = px.area(h, y="Close", title="2 Anos")
                            fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                except: st.write("Gráfico indisponível.")
            
            st.metric("Score", f"{score}/10")
            st.info(txt)
            
            with st.expander("Mais Dados"):
                st.write(f"Dívida/PL: {row.get('Div_Patrimonio',0)}")
                st.write(f"Margem: {row.get('MargemLiquida',0):.1f}%")

        else:
            st.info("👈 Selecione um ativo.")

else:
    st.error("Fundamentus offline. Tente mais tarde.")
