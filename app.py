import streamlit as st
import pandas as pd
import fundamentus # A nossa nova arma secreta
import yfinance as yf
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Dados Oficiais", layout="wide")

st.title("🏆 Investidor Pro 3.0: Dados Oficiais")
st.markdown("Monitor de mercado com dados extraídos diretamente da base Fundamentus (B3).")

# --- BARRA LATERAL ---
st.sidebar.header("Filtros de Carteira")
opcao_carteira = st.sidebar.radio("Selecione o Universo:", ["Carteira Recomendada", "Top 10 Dividendos (Geral)"])

# Carteira Teórica (Sem o .SA, pois o padrão brasileiro não usa)
meus_ativos = ['BBAS3', 'TAEE11', 'VALE3', 'ITSA4', 'CPLE6', 'KLBN11', 
               'MXRF11', 'HGLG11', 'KNRI11', 'VISC11']

# --- MOTOR DE DADOS HÍBRIDO ---
@st.cache_data
def buscar_dados_fundamentus():
    """Baixa a tabela completa de ativos da bolsa brasileira."""
    try:
        # get_resultado() já retorna P/L, P/VP, DY, etc. limpos
        df = fundamentus.get_resultado()
        
        # Resetando o index para o Ticker virar coluna
        df = df.reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Filtros de limpeza (remover liquidez zerada)
        df = df[df['Liq.2meses'] > 100000] 
        
        return df
    except Exception as e:
        st.error(f"Erro ao conectar com Fundamentus: {e}")
        return pd.DataFrame()

# --- INTERFACE ---
if st.button('🔄 Atualizar Base de Dados (B3)'):
    with st.spinner('Conectando ao servidor de dados...'):
        df_completo = buscar_dados_fundamentus()
        
        if not df_completo.empty:
            # Lógica de Seleção
            if opcao_carteira == "Carteira Recomendada":
                # Filtra apenas os ativos da nossa lista
                df_final = df_completo[df_completo['Ticker'].isin(meus_ativos)].copy()
            else:
                # Top 10 maiores pagadores da bolsa (com filtro de liquidez)
                df_final = df_completo.sort_values(by='Div.Yield', ascending=False).head(10).copy()

            # --- TRATAMENTO DE DADOS ---
            # O Fundamentus retorna decimais (ex: 0.12 para 12%), vamos ajustar para leitura humana
            cols_percent = ['Div.Yield', 'Mrg. Líq.']
            for col in cols_percent:
                if col in df_final.columns:
                    df_final[col] = df_final[col] * 100

            # Selecionar apenas colunas úteis para o Dashboard
            colunas_visuais = ['Ticker', 'Cotação', 'P/L', 'P/VP', 'Div.Yield', 'EV/EBIT', 'ROE']
            df_display = df_final[colunas_visuais].sort_values(by='Div.Yield', ascending=False)
            
            # Renomear para ficar bonito
            df_display.columns = ['Ticker', 'Preço (R$)', 'P/L (Anos)', 'P/VP', 'DY (%)', 'EV/EBIT', 'ROE (%)']

            # --- DASHBOARD VISUAL ---
            
            # 1. KPIs do Topo
            col1, col2, col3 = st.columns(3)
            melhor_ativo = df_display.iloc[0]
            
            col1.metric("Campeão de Renda", melhor_ativo['Ticker'], f"{melhor_ativo['DY (%)']:.2f}%")
            col2.metric("P/VP Médio da Carteira", f"{df_display['P/VP'].mean():.2f}")
            col3.metric("Ativos Monitorados", len(df_display))
            
            st.divider()

            # 2. Gráfico de Comparação
            col_graf, col_tab = st.columns([1, 1])
            
            with col_graf:
                st.subheader("Quem está mais barato? (P/VP)")
                # Gráfico de barras horizontal para P/VP
                fig = px.bar(df_display, x='P/VP', y='Ticker', orientation='h', 
                             color='P/VP', color_continuous_scale='RdYlGn_r', # Verde se for baixo (barato)
                             title="Ranking de Preço (Quanto menor, melhor)")
                # Adiciona linha de referência (Preço Justo = 1.0)
                fig.add_vline(x=1, line_dash="dash", line_color="white", annotation_text="Preço Justo")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_tab:
                st.subheader("Relatório Fundamentalista")
                st.dataframe(
                    df_display.style.format("{:.2f}", subset=['Preço (R$)', 'P/L (Anos)', 'P/VP', 'DY (%)', 'ROE (%)'])
                                    .highlight_between(left=0, right=1.0, subset=['P/VP'], color='lightgreen'), # P/VP abaixo de 1 fica verde
                    use_container_width=True,
                    height=400
                )

            # --- DETALHES DE UM ATIVO (Gráfico Yahoo) ---
            st.divider()
            st.subheader("🔎 Raio-X Histórico")
            ativo_select = st.selectbox("Selecione para ver o gráfico de preço:", df_display['Ticker'])
            
            if ativo_select:
                ticker_yahoo = ativo_select + ".SA" # Adiciona sufixo para o Yahoo achar
                try:
                    hist = yf.Ticker(ticker_yahoo).history(period="1y")
                    fig_line = px.line(hist, y='Close', title=f"Evolução de Preço: {ativo_select} (1 Ano)")
                    st.plotly_chart(fig_line, use_container_width=True)
                except:
                    st.warning("Gráfico histórico indisponível no momento.")

else:
    st.info("Clique no botão para baixar os dados oficiais da B3.")
