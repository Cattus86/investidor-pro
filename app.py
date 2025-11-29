import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import fundamentus

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Dashboard", layout="wide")

st.title("🏆 Investidor Pro: Monitor de Mercado")
st.markdown("Monitor de mercado com inteligência de dados híbrida (Fundamentus + Yahoo).")

# --- BARRA LATERAL ---
st.sidebar.header("Filtros")
opcao_carteira = st.sidebar.radio("Visualizar:", ["Minha Carteira", "Top Oportunidades"])

# Lista de ativos padrão
meus_ativos = ['BBAS3', 'TAEE11', 'VALE3', 'ITSA4', 'CPLE6', 'KLBN11', 
               'MXRF11', 'HGLG11', 'KNRI11', 'VISC11']

# --- FUNÇÕES DE COLETA DE DADOS ---

def buscar_dados_yfinance_backup(lista_ativos):
    """
    MODO DE CONTINGÊNCIA:
    Se o Fundamentus falhar, usamos o Yahoo Finance para não deixar o site vazio.
    """
    dados = []
    for ticker in lista_ativos:
        try:
            # Adiciona .SA para o Yahoo
            ticker_yahoo = ticker if ".SA" in ticker else f"{ticker}.SA"
            info = yf.Ticker(ticker_yahoo).info
            
            # Coleta segura
            preco = info.get('currentPrice') or info.get('regularMarketPrice') or 0
            dy = info.get('dividendYield', 0) * 100
            pvp = info.get('priceToBook', 0)
            pl = info.get('trailingPE', 0)
            roe = info.get('returnOnEquity', 0) * 100
            
            dados.append({
                'Ticker': ticker.replace(".SA", ""),
                'Preço (R$)': preco,
                'DY (%)': dy,
                'P/VP': pvp,
                'P/L': pl,
                'ROE (%)': roe,
                'Fonte': 'Yahoo (Backup)'
            })
        except:
            pass
    return pd.DataFrame(dados)

@st.cache_data(ttl=3600) # Cache de 1 hora
def buscar_dados_hibridos():
    """
    Tenta pegar do Fundamentus (Oficial). 
    Se der erro, ativa o plano B (Yahoo).
    """
    try:
        # TENTATIVA 1: Fundamentus
        df = fundamentus.get_resultado_raw() # Tenta pegar dados brutos
        
        # Se retornou vazio, levanta erro para ir pro except
        if df.empty:
            raise Exception("Retorno vazio do Fundamentus")

        df = df.reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Tratamento de Colunas (Onde deu o erro antes)
        # Vamos usar apenas colunas que temos certeza que existem ou tratá-las
        cols_map = {
            'Cotação': 'Preço (R$)',
            'Div.Yield': 'DY (%)',
            'P/VP': 'P/VP',
            'P/L': 'P/L',
            'ROE': 'ROE (%)',
            'Liq.2meses': 'Liquidez'
        }
        
        # Renomeia o que encontrar
        df = df.rename(columns=cols_map)
        
        # Filtro de Liquidez (Seguro)
        if 'Liquidez' in df.columns:
            df = df[df['Liquidez'] > 100000]
        
        # Ajuste de Porcentagem (0.12 -> 12.0)
        cols_percent = ['DY (%)', 'ROE (%)']
        for col in cols_percent:
            if col in df.columns:
                df[col] = df[col] * 100
                
        df['Fonte'] = 'Fundamentus (Oficial)'
        return df

    except Exception as e:
        # TENTATIVA 2: Plano B (Yahoo Finance)
        # st.warning(f"Nota: Usando dados alternativos devido a instabilidade na fonte oficial. Erro: {e}")
        return buscar_dados_yfinance_backup(meus_ativos)

# --- INTERFACE ---
if st.button('🔄 Atualizar Dados'):
    with st.spinner('Analisando o mercado...'):
        df_final = buscar_dados_hibridos()
        
        if not df_final.empty:
            
            # Filtros de visualização
            if opcao_carteira == "Minha Carteira":
                df_display = df_final[df_final['Ticker'].isin(meus_ativos)].copy()
            else:
                df_display = df_final.sort_values(by='DY (%)', ascending=False).head(15).copy()
            
            # --- DASHBOARD ---
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            top_asset = df_display.sort_values(by='DY (%)', ascending=False).iloc[0]
            
            col1.metric("Maior Pagador", top_asset['Ticker'], f"{top_asset['DY (%)']:.2f}%")
            col2.metric("Média P/VP", f"{df_display['P/VP'].mean():.2f}")
            col3.caption(f"Fonte dos Dados: {df_display['Fonte'].iloc[0]}")
            
            st.markdown("---")
            
            # Gráficos e Tabelas
            col_graf, col_tab = st.columns([1, 1])
            
            with col_graf:
                fig = px.bar(df_display, x='Ticker', y='DY (%)', color='P/VP',
                             title="Dividend Yield x Preço (Cor = P/VP)",
                             color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
                
            with col_tab:
                # Formatação da Tabela
                st.dataframe(
                    df_display[['Ticker', 'Preço (R$)', 'DY (%)', 'P/VP', 'P/L']].style
                    .format("{:.2f}", subset=['Preço (R$)', 'DY (%)', 'P/VP', 'P/L'])
                    .highlight_max(subset=['DY (%)'], color='lightgreen'),
                    use_container_width=True,
                    height=400
                )
                
        else:
            st.error("Não foi possível conectar a nenhuma fonte de dados no momento.")
else:
    st.info("Clique no botão para carregar o painel.")
