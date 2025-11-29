import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURAÇÃO DA PÁGINA E CSS PROFISSIONAL ---
st.set_page_config(page_title="Investidor Pro | Platinum", layout="wide", initial_sidebar_state="expanded")

# CSS Customizado para visual de terminal financeiro
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #00c853; /* Verde financeiro */
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 5px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00c853 !important;
        color: white !important;
    }
    /* Destaque para o painel lateral de detalhes */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #131722;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2a2e39;
    }
</style>
""", unsafe_allow_html=True)

st.title("💎 Investidor Pro: Platinum Edition")
st.markdown("##### Plataforma de Análise Quantitativa Avançada")

# --- 2. FUNÇÕES CORE (DADOS E CÁLCULOS - Mantidas da versão anterior) ---
def limpar_numero_ptbr(valor):
    if isinstance(valor, str):
        valor_limpo = valor.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(valor_limpo)
        except: return 0.0
    return float(valor) if valor else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def carregar_dados_platinum():
    try:
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
            if col != 'Ticker': df[col] = df[col].apply(limpar_numero_ptbr)
        if 'DY' in df.columns and df['DY'].mean() < 1: df['DY'] *= 100
        if 'ROE' in df.columns and df['ROE'].mean() < 1: df['ROE'] *= 100
        if 'MargemLiquida' in df.columns and df['MargemLiquida'].mean() < 1: df['MargemLiquida'] *= 100
        return df
    except Exception as e:
        st.error(f"Erro crítico: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    # Graham
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    mask_valida = (df['LPA'] > 0) & (df['VPA'] > 0)
    df.loc[mask_valida, 'Graham_Valor'] = np.sqrt(22.5 * df.loc[mask_valida, 'LPA'] * df.loc[mask_valida, 'VPA'])
    df['Graham_Valor'] = df['Graham_Valor'].fillna(0)
    df['Graham_Upside'] = np.where((df['Graham_Valor'] > 0) & (df['Preco'] > 0), ((df['Graham_Valor'] - df['Preco']) / df['Preco']) * 100, -999)
    # Magic Formula
    df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
    if not df_magic.empty:
        df_magic['R_EV'] = df_magic['EV_EBIT'].rank(ascending=True)
        df_magic['R_ROIC'] = df_magic['ROIC'].rank(ascending=False)
        df_magic['Score_Magic'] = df_magic['R_EV'] + df_magic['R_ROIC']
        df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
    else:
        df['Score_Magic'] = 99999
    # Bazin
    df['Bazin_Teto'] = np.where(df['DY'] > 0, df['Preco'] * (df['DY'] / 6), 0)
    return df

# --- 3. ROBÔ ANALISTA ---
def analisar_ativo(row):
    score = 5
    texto = []
    # Dividendos
    if row['DY'] > 10: texto.append("🟢 **Yield:** Altíssimo (>10%). Verificar recorrencia."); score += 2
    elif row['DY'] > 6: texto.append("🔵 **Yield:** Atrativo (>6%)."); score += 1
    else: texto.append("⚪ **Yield:** Baixo."); score -= 1
    # Preço
    if row['PL'] <= 0: texto.append("🔴 **P/L:** Empresa com prejuízo."); score -= 3
    elif row['PL'] < 5: texto.append("🟢 **P/L:** Muito descontada (<5)."); score += 2
    elif row['PL'] < 15: texto.append("🔵 **P/L:** Razoável."); score += 1
    # Qualidade
    if row['ROE'] > 15: texto.append("🔥 **ROE:** Alta rentabilidade (>15%)."); score += 2
    elif row['ROE'] < 5: texto.append("❄️ **ROE:** Rentabilidade baixa."); score -= 1
    # Graham
    if row['Graham_Upside'] > 20: texto.append(f"💎 **Graham:** Potencial teórico de +{row['Graham_Upside']:.0f}%."); score += 1
    
    return score, "\n\n".join(texto)

# --- 4. INTERFACE PRINCIPAL ---

with st.spinner('Sincronizando dados do mercado...'):
    df_raw = carregar_dados_platinum()

if not df_raw.empty:
    df = calcular_indicadores(df_raw)
    
    # --- SIDEBAR (FILTROS) ---
    st.sidebar.header("🔍 Filtros Avançados")
    busca = st.sidebar.text_input("Buscar Ticker:", placeholder="Ex: VALE3").upper().strip()
    liq_min = st.sidebar.select_slider("Liquidez Diária Mínima:", options=[0, 50000, 200000, 1000000, 5000000], value=200000)
    df_view = df[df['Liquidez'] >= liq_min].copy()
    if busca: df_view = df_view[df_view['Ticker'].str.contains(busca)]
    
    st.sidebar.markdown("---")
    st.sidebar.download_button("📥 Baixar Dados (Excel/CSV)", df_view.to_csv(sep=';', decimal=','), "relatorio_platinum.csv", "text/csv")

    # --- DASHBOARD EXECUTIVO (TOPO) ---
    # Médias do mercado para dar contexto
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Ativos Analisados", len(df_view))
    col_m2.metric("Média de Dividend Yield", f"{df_view[df_view['DY']>0]['DY'].mean():.2f}%")
    col_m3.metric("Média P/L do Mercado", f"{df_view[(df_view['PL']>0) & (df_view['PL']<100)]['PL'].mean():.1f}x")
    col_m4.metric("Liquidez Total Diária", f"R$ {df_view['Liquidez'].sum()/1_000_000_000:.1f} Bi")
    st.divider()

    # --- LAYOUT PRINCIPAL (2 COLUNAS: ESQUERDA=DADOS, DIREITA=DETALHES) ---
    col_main, col_details = st.columns([2.2, 1]) # Coluna da direita mais estreita para o resumo

    # Inicializa estado da seleção
    if 'ativo_selecionado' not in st.session_state: st.session_state['ativo_selecionado'] = None

    # Função de clique
    def on_click_tabela(evento, dataframe_origem):
        if len(evento.selection.rows) > 0:
            idx = evento.selection.rows[0]
            st.session_state['ativo_selecionado'] = dataframe_origem.iloc[idx]

    # Configuração Visual Comum das Tabelas
    cfg = {
        "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
        "DY": st.column_config.ProgressColumn("Yield (DY)", format="%.2f%%", min_value=0, max_value=15),
        "PVP": st.column_config.NumberColumn("P/VP", format="%.2f"),
        "PL": st.column_config.NumberColumn("P/L", format="%.2f"),
        "Graham_Upside": st.column_config.ProgressColumn("Potencial Graham", format="%.0f%%", min_value=0, max_value=100),
        "Bazin_Teto": st.column_config.NumberColumn("Teto Bazin (6%)", format="R$ %.2f"),
        "Score_Magic": st.column_config.NumberColumn("Score Magic", format="%d"),
        "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%")
    }

    with col_main:
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Dividendos", "💎 Graham", "✨ Magic Formula", "📈 Analytics (Gráficos)"])
        
        with tab1:
            df_t1 = df_view.nlargest(100, 'DY')
            st.caption("Selecione uma linha para ver a análise detalhada.")
            ev1 = st.dataframe(df_t1[['Ticker', 'Preco', 'DY', 'PVP', 'Bazin_Teto', 'ROE']], 
                             column_config=cfg, hide_index=True, use_container_width=True, 
                             on_select="rerun", selection_mode="single-row", height=500)
            on_click_tabela(ev1, df_t1)

        with tab2:
            df_t2 = df_view[(df_view['Graham_Upside'] > 0) & (df_view['Graham_Upside'] < 500)].nlargest(100, 'Graham_Upside')
            st.caption("Selecione uma linha para ver a análise detalhada.")
            ev2 = st.dataframe(df_t2[['Ticker', 'Preco', 'Graham_Valor', 'Graham_Upside', 'PL', 'PVP']], 
                             column_config={"Graham_Valor": st.column_config.NumberColumn("Valor Justo", format="R$ %.2f"), **cfg}, 
                             hide_index=True, use_container_width=True, 
                             on_select="rerun", selection_mode="single-row", height=500)
            on_click_tabela(ev2, df_t2)

        with tab3:
            df_t3 = df_view.nsmallest(100, 'Score_Magic')
            st.caption("Selecione uma linha para ver a análise detalhada.")
            ev3 = st.dataframe(df_t3[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']], 
                             column_config=cfg, hide_index=True, use_container_width=True, 
                             on_select="rerun", selection_mode="single-row", height=500)
            on_click_tabela(ev3, df_t3)

        with tab4:
            st.subheader("Visualização de Mercado")
            
            # GRÁFICO 1: TOP DIVIDENDOS (BARRAS HORIZONTAIS)
            top_dy_chart = df_view.nlargest(15, 'DY').sort_values(by='DY', ascending=True)
            fig_bar = px.bar(top_dy_chart, x='DY', y='Ticker', orientation='h', 
                             title="Top 15 Maiores Dividend Yields", text_auto='.2f',
                             color='DY', color_continuous_scale='greens')
            fig_bar.update_layout(xaxis_title="Dividend Yield (%)", yaxis_title="", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                # GRÁFICO 2: MAPA DE CALOR (Existing Scatter)
                df_chart = df_view[(df_view['PL'] > 0) & (df_view['PL'] < 50) & (df_view['ROE'] > 0) & (df_view['ROE'] < 60)]
                fig_scatter = px.scatter(df_chart, x='PL', y='ROE', size='Liquidez', color='DY', 
                                 hover_name='Ticker', title="Risco (P/L) x Retorno (ROE)", 
                                 color_continuous_scale='RdYlGn', height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col_g2:
                # GRÁFICO 3: HISTOGRAMA DE P/L (NOVO)
                df_hist = df_view[(df_view['PL'] > -20) & (df_view['PL'] < 50)]
                fig_hist = px.histogram(df_hist, x="PL", nbins=30, title="Distribuição de P/L do Mercado",
                                        color_discrete_sequence=['#00c853'], height=400)
                fig_hist.update_layout(xaxis_title="P/L (Anos)", yaxis_title="Quantidade de Empresas")
                fig_hist.add_vline(x=df_hist['PL'].median(), line_dash="dash", line_color="white", annotation_text="Mediana")
                st.plotly_chart(fig_hist, use_container_width=True)

    # --- PAINEL LATERAL DIREITO (DETALHES DA AÇÃO) ---
    with col_details:
        st.subheader("📊 Resumo do Ativo")
        
        if st.session_state['ativo_selecionado'] is not None:
            row = st.session_state['ativo_selecionado']
            score, texto = analisar_ativo(row)
            
            # Cabeçalho do Ativo
            st.markdown(f"# {row['Ticker']}")
            st.metric("Preço Atual", f"R$ {row['Preco']:.2f}")
            
            st.divider()
            
            # Score e Análise do Robô
            col_score, col_txt = st.columns([1, 3])
            with col_score:
                st.metric("Robô Score", f"{score}/10")
                if score >= 7: st.success("Excelente")
                elif score >= 4: st.warning("Neutro")
                else: st.error("Cuidado")
                
            st.markdown("### 🤖 Análise Rápida")
            st.info(texto)
            
            # Mini Gráfico Comparativo (NOVO)
            st.divider()
            st.markdown("### Comparativo de Yield")
            media_mercado_dy = df_view[df_view['DY']>0]['DY'].mean()
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = row['DY'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Dividend Yield vs Média Mercado"},
                delta = {'reference': media_mercado_dy, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, max(20, row['DY'] + 5)]},
                    'bar': {'color': "#00c853"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75, 'value': media_mercado_dy
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        else
