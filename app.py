import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# --- 1. CONFIGURAÇÃO VISUAL PROFISSIONAL ---
st.set_page_config(page_title="Investidor Pro | Titanium", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Estilo Dark Mode Profissional */
    .stApp { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; color: #00c853; }
    /* Abas customizadas */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937; color: #e5e7eb; border-radius: 4px; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #00c853 !important; color: white !important; }
    /* Painel Lateral */
    section[data-testid="stSidebar"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Investidor Pro: Titanium Edition")
st.markdown("##### Plataforma Quantitativa: Fundamentalismo + Momentum + Setores")

# --- 2. MAPA DE SETORES ---
MAPA_SETORES = {
    'Bancos': ['BBAS3', 'ITUB4', 'BBDC4', 'SANB11', 'BPAC11', 'ABCB4', 'BRSR6', 'ITSA4', 'BBSE3', 'CXSE3'],
    'Energia': ['PETR4', 'PETR3', 'PRIO3', 'VBBR3', 'UGPA3', 'CSAN3', 'ENAT3', 'RRRP3', 'RECV3'],
    'Elétricas': ['ELET3', 'ELET6', 'EGIE3', 'TRPL4', 'TAEE11', 'CPLE6', 'CMIG4', 'EQTL3', 'LIGT3', 'NEOE3', 'ALUP11'],
    'Mineração/Sid': ['VALE3', 'CSNA3', 'GGBR4', 'GOAU4', 'USIM5', 'CMIN3', 'FESA4'],
    'Varejo': ['MGLU3', 'LREN3', 'ARZZ3', 'SOMA3', 'PETZ3', 'RDOR3', 'RADL3', 'AMER3', 'BHIA3'],
    'Bens Ind.': ['WEGE3', 'EMBR3', 'TUPY3', 'RAPT4', 'POMO4'],
    'Construção': ['CYRE3', 'EZTC3', 'MRVE3', 'TEND3', 'JHSF3'],
    'Saneamento': ['SBSP3', 'CSMG3', 'SAPR11', 'SAPR4'],
    'Agronegócio': ['SLCE3', 'AGRO3', 'SOJA3', 'TTEN3'],
    'Seguros': ['BBSE3', 'CXSE3', 'PSSA3', 'SULA11'],
    'Imobiliário': ['MULT3', 'IGTI11', 'ALSO3']
}

def obter_setor(ticker):
    ticker_clean = ticker.upper().strip()
    for setor, lista in MAPA_SETORES.items():
        if ticker_clean in lista:
            return setor
    return "Outros / Geral"

# --- 3. MOTOR DE DADOS HÍBRIDO ---
def limpar_numero(valor):
    if isinstance(valor, str):
        valor = valor.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(valor)
        except: return 0.0
    return float(valor) if valor else 0.0

@st.cache_data(ttl=600, show_spinner=False)
def carregar_dados_completo():
    try:
        # A. DADOS FUNDAMENTALISTAS
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        mapa = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio'
        }
        cols = ['Ticker'] + [c for c in mapa.keys() if c in df.columns]
        df = df[cols].copy()
        df.rename(columns=mapa, inplace=True)
        
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(limpar_numero)
            
        for col in ['DY', 'ROE', 'MargemLiquida']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # B. ENRIQUECIMENTO
        df['Setor'] = df['Ticker'].apply(obter_setor)

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

def calcular_momentum_lote(df_filtrado):
    tickers = [t + ".SA" for t in df_filtrado['Ticker'].tolist()]
    if not tickers: return df_filtrado
    
    try:
        # Baixa histórico em lote
        dados_hist = yf.download(tickers, period="6mo", progress=False)['Adj Close']
        
        if not dados_hist.empty:
            momentum_dict = {}
            # Itera sobre as colunas (que são os tickers)
            # Se for apenas um ticker, dados_hist pode ser uma Series, não DataFrame
            if isinstance(dados_hist, pd.Series):
                dados_hist = dados_hist.to_frame(name=tickers[0])

            for t in dados_hist.columns: 
                try:
                    serie = dados_hist[t].dropna()
                    if len(serie) > 10:
                        preco_ini = serie.iloc[0]
                        preco_fim = serie.iloc[-1]
                        retorno = ((preco_fim - preco_ini) / preco_ini) * 100
                        # Remove o .SA para mapear de volta
                        ticker_key = t.replace('.SA', '')
                        momentum_dict[ticker_key] = retorno
                except:
                    pass
            
            df_filtrado['Momentum_6M'] = df_filtrado['Ticker'].map(momentum_dict).fillna(0)
    except:
        df_filtrado['Momentum_6M'] = 0.0
        
    return df_filtrado

def calcular_kpis(df):
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

# --- 4. ANALISTA AUTOMÁTICO ---
def analisar(row):
    score = 5
    txt = []
    
    # Fundamentalismo
    if row['DY'] > 9: txt.append("🟢 **Yield:** Excelente (>9%)."); score += 2
    elif row['DY'] < 3: txt.append("⚪ **Yield:** Baixo."); score -= 1
    
    if row['PL'] < 5 and row['PL'] > 0: txt.append("🟢 **P/L:** Muito descontado."); score += 2
    elif row['PL'] > 20: txt.append("🔴 **P/L:** Preço esticado."); score -= 1
    
    if row['ROE'] > 15: txt.append("🔥 **ROE:** Alta rentabilidade."); score += 2
    
    # Momentum
    mom = row.get('Momentum_6M', 0)
    if mom > 20: txt.append(f"🚀 **Momentum:** Forte tendência de alta (+{mom:.1f}% em 6m)."); score += 2
    elif mom < -10: txt.append(f"🐻 **Momentum:** Tendência de baixa ({mom:.1f}% em 6m)."); score -= 2
    
    return score, "\n\n".join(txt)

# --- 5. INTERFACE ---
with st.spinner('Carregando dados da B3...'):
    df_full = carregar_dados_completo()

if not df_full.empty:
    df = calcular_kpis(df_full)
    
    # --- SIDEBAR (FILTROS) ---
    st.sidebar.header("🔍 Filtros & Setores")
    
    setores_disponiveis = ["Todos"] + sorted(list(set(df['Setor'].unique())))
    setor_selecionado = st.sidebar.selectbox("Filtrar por Setor:", setores_disponiveis)
    
    # CORREÇÃO DO ERRO DE VALUE ERROR:
    # A lista de options DEVE conter o valor padrão (200000)
    liq_min = st.sidebar.select_slider(
        "Liquidez Mínima:", 
        options=[0, 50000, 200000, 1000000, 5000000], 
        value=200000
    )
    
    # Aplicação dos Filtros
    df_view = df[df['Liquidez'] >= liq_min].copy()
    if setor_selecionado != "Todos":
        df_view = df_view[df_view['Setor'] == setor_selecionado]
        
    # --- CÁLCULO DE MOMENTUM (ON-DEMAND) ---
    with st.spinner('Calculando Momentum (Yahoo Finance)...'):
        # Limita a 80 papéis para não estourar o tempo de requisição
        if len(df_view) > 80:
            # Prioriza as mais líquidas para calcular momentum
            df_calc = df_view.nlargest(80, 'Liquidez')
            # Mantém as outras sem momentum calculado (0)
            df_resto = df_view[~df_view['Ticker'].isin(df_calc['Ticker'])]
            
            df_calc = calcular_momentum_lote(df_calc)
            df_view = pd.concat([df_calc, df_resto])
        else:
            df_view = calcular_momentum_lote(df_view)

    # --- DASHBOARD KPI ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ações Filtradas", len(df_view))
    
    if len(df_view) > 0:
        c2.metric("Yield Médio", f"{df_view[df_view['DY']>0]['DY'].mean():.2f}%")
        c3.metric("P/L Médio", f"{df_view[(df_view['PL']>0)&(df_view['PL']<50)]['PL'].mean():.1f}x")
        try:
            top_setor = df_view.groupby('Setor')['Momentum_6M'].mean().idxmax()
            mom_setor = df_view.groupby('Setor')['Momentum_6M'].mean().max()
            c4.metric(f"Setor Quente: {top_setor}", f"+{mom_setor:.1f}%")
        except:
            c4.metric("Setor Quente", "N/A")
    else:
        st.warning("Nenhuma ação encontrada com esses filtros.")

    st.divider()

    # --- ÁREA PRINCIPAL ---
    col_table, col_detail = st.columns([2, 1])
    
    if 'sel' not in st.session_state: st.session_state['sel'] = None
    
    def on_sel(evt, df_ref):
        if len(evt.selection.rows)>0:
            # Como o df visual pode estar filtrado/ordenado, usamos iloc no df de referencia
            st.session_state['sel'] = df_ref.iloc[evt.selection.rows[0]]

    cfg = {
        "Preco": st.column_config.NumberColumn("R$", format="R$ %.2f"),
        "Momentum_6M": st.column_config.ProgressColumn("Momentum (6m)", format="%.1f%%", min_value=-30, max_value=30),
        "DY": st.column_config.ProgressColumn("Yield", format="%.1f%%", min_value=0, max_value=15),
        "Graham": st.column_config.NumberColumn("Preço Justo", format="R$ %.2f"),
        "Score_Magic": st.column_config.NumberColumn("Score", format="%d")
    }

    with col_table:
        t1, t2, t3, t4 = st.tabs(["🚀 Momentum", "💰 Dividendos", "💎 Valor", "📈 Setores"])
        
        with t1:
            st.caption("Ações com maior valorização recente (Tendência).")
            df_mom = df_view.sort_values(by='Momentum_6M', ascending=False).head(50)
            ev = st.dataframe(df_mom[['Ticker', 'Setor', 'Preco', 'Momentum_6M', 'PL']], 
                         column_config=cfg, hide_index=True, use_container_width=True, 
                         on_select="rerun", selection_mode="single-row", height=450)
            on_sel(ev, df_mom)
            
        with t2:
            df_div = df_view.nlargest(50, 'DY')
            ev = st.dataframe(df_div[['Ticker', 'Setor', 'Preco', 'DY', 'Bazin']], 
                         column_config=cfg, hide_index=True, use_container_width=True, 
                         on_select="rerun", selection_mode="single-row", height=450)
            on_sel(ev, df_div)
            
        with t3:
            df_val = df_view[(df_view['Upside_Graham']>0) & (df_view['Upside_Graham']<500)].nlargest(50, 'Upside_Graham')
            ev = st.dataframe(df_val[['Ticker', 'Preco', 'Graham', 'Upside_Graham', 'PL']], 
                         column_config={"Upside_Graham": st.column_config.NumberColumn("Potencial %", format="%.1f%%"), **cfg}, 
                         hide_index=True, use_container_width=True, 
                         on_select="rerun", selection_mode="single-row", height=450)
            on_sel(ev, df_val)

        with t4:
            st.subheader("Performance por Setor")
            if len(df_view) > 0:
                df_setor = df_view.groupby('Setor')[['DY', 'PL', 'Momentum_6M']].mean().reset_index()
                fig_bar = px.bar(df_setor, x='Momentum_6M', y='Setor', orientation='h', title="Momentum Médio", color='Momentum_6M', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_bar, use_container_width=True)

    # --- PAINEL DETALHES ---
    with col_detail:
        st.markdown("### 📊 Raio-X do Ativo")
        if st.session_state['sel'] is not None:
            row = st.session_state['sel']
            score, txt = analisar(row)
            
            st.markdown(f"# {row['Ticker']}")
            st.caption(f"Setor: {row['Setor']}")
            
            c_p, c_m = st.columns(2)
            c_p.metric("Preço", f"R$ {row['Preco']:.2f}")
            
            # Tratamento seguro para momentum se não existir
            mom_val = row.get('Momentum_6M', 0)
            c_m.metric("Tendência (6m)", f"{mom_val:.1f}%", delta_color="normal")
            
            st.divider()
            st.metric("Score Robô", f"{score}/10")
            st.info(txt)
            
            st.divider()
            
            # Gráfico Histórico
            try:
                with st.spinner('Baixando gráfico...'):
                    # Adiciona .SA para Yahoo Finance
                    t_yahoo = row['Ticker']
                    if not t_yahoo.endswith('.SA'): t_yahoo += ".SA"
                    
                    hist = yf.Ticker(t_yahoo).history(period="1y")
                    if not hist.empty:
                        fig_line = px.area(hist, y="Close", title="Preço (1 Ano)")
                        fig_line.update_layout(showlegend=False, margin=dict(l=0,r=0,t=30,b=0), height=200)
                        fig_line.update_xaxes(visible=False)
                        st.plotly_chart(fig_line, use_container_width=True)
                    else:
                        st.warning("Sem dados históricos recentes.")
            except:
                st.write("Gráfico indisponível.")
                
        else:
            st.info("👈 Selecione uma ação na tabela para ver a análise.")

else:
    st.error("Erro ao conectar.")
