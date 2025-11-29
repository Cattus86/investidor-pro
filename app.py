import streamlit as st
import pandas as pd
import fundamentus
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Quant", layout="wide")
st.title("⚡ Investidor Pro: Plataforma Quantitativa")
st.markdown("Análise fundamentalista automatizada de todos os ativos da B3.")

# --- BARRA LATERAL ---
st.sidebar.header("🔍 Filtros Globais")
min_liquidez = st.sidebar.number_input("Liquidez Diária Mínima (R$):", value=200000, step=50000)

# --- MOTOR DE DADOS BLINDADO ---
@st.cache_data(ttl=3600)
def carregar_base_completa():
    try:
        # 1. Baixar dados brutos
        df = fundamentus.get_resultado()
        df = df.reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # 2. PADRONIZAÇÃO DE COLUNAS (A Correção do Erro)
        # Vamos renomear tudo para nomes simples (sem pontos ou barras)
        # Isso evita o erro de 'KeyError: P/L'
        mapa_colunas = {
            'Cotação': 'Preco',
            'P/L': 'PL',           # Removemos a barra /
            'P/VP': 'PVP',         # Removemos a barra /
            'Div.Yield': 'DY',
            'ROE': 'ROE',
            'ROIC': 'ROIC',
            'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez',
            'Mrg. Líq.': 'MargemLiquida'
        }
        
        # Renomeia apenas o que encontrar
        df = df.rename(columns=mapa_colunas)
        
        # 3. Tratamento de Tipos (Texto -> Número)
        colunas_percentuais = ['DY', 'ROE', 'ROIC', 'MargemLiquida']
        for col in colunas_percentuais:
            if col in df.columns:
                # O Fundamentus às vezes manda 0.15, transformamos em 15.0
                df[col] = df[col] * 100

        # Garantir que PL e PVP sejam números (tratando erros de texto)
        cols_numericas = ['Preco', 'PL', 'PVP', 'EV_EBIT', 'Liquidez']
        for col in cols_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 4. CÁLCULO DE GRAHAM (Usando os nomes novos PL e PVP)
        # LPA = Preço / PL
        # VPA = Preço / PVP
        
        # Evita divisão por zero
        df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
        df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
        
        def calcular_graham(row):
            # Fórmula: Raiz(22.5 * LPA * VPA)
            if row['LPA'] > 0 and row['VPA'] > 0:
                return np.sqrt(22.5 * row['LPA'] * row['VPA'])
            return 0
            
        df['Preco_Justo_Graham'] = df.apply(calcular_graham, axis=1)
        
        # Potencial de Valorização
        df['Potencial_Graham'] = np.where(
            (df['Preco_Justo_Graham'] > 0) & (df['Preco'] > 0),
            ((df['Preco_Justo_Graham'] - df['Preco']) / df['Preco']) * 100,
            -999
        )

        # 5. CÁLCULO MAGIC FORMULA (EV_EBIT + ROIC)
        if 'EV_EBIT' in df.columns and 'ROIC' in df.columns:
            # Filtra dados válidos para o ranking
            df_magic = df[(df['EV_EBIT'] > 0) & (df['ROIC'] > 0)].copy()
            
            df_magic['Rank_EV_EBIT'] = df_magic['EV_EBIT'].rank(ascending=True)
            df_magic['Rank_ROIC'] = df_magic['ROIC'].rank(ascending=False)
            df_magic['Score_Magic'] = df_magic['Rank_EV_EBIT'] + df_magic['Rank_ROIC']
            
            # Mescla de volta
            df = df.merge(df_magic[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999 # Se não tiver dados, joga pro fim da fila

        return df

    except Exception as e:
        st.error(f"Erro detalhado no processamento: {e}")
        # Retorna dataframe vazio mas com colunas para não quebrar a tela
        return pd.DataFrame(columns=['Ticker', 'Preco', 'DY', 'PL', 'PVP', 'Liquidez'])

# --- CARREGAMENTO ---
with st.spinner('Processando Big Data da B3...'):
    df_raw = carregar_base_completa()

if not df_raw.empty and 'Liquidez' in df_raw.columns:
    # Filtro de Liquidez
    df = df_raw[df_raw['Liquidez'] >= min_liquidez].copy()
    
    # --- INTERFACE (TABS) ---
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Visão Geral", "💰 Dividendos", "⚖️ Graham", "✨ Magic Formula"])

    with tab1:
        st.subheader("Mercado Completo")
        colunas_mostrar = ['Ticker', 'Preco', 'PL', 'PVP', 'DY', 'ROE', 'MargemLiquida']
        # Filtra colunas que realmente existem
        cols_final = [c for c in colunas_mostrar if c in df.columns]
        st.dataframe(df[cols_final].set_index('Ticker'), use_container_width=True)

    with tab2:
        st.subheader("Top Dividendos")
        if 'DY' in df.columns:
            df_div = df.sort_values(by='DY', ascending=False).head(20)
            st.dataframe(
                df_div[['Ticker', 'Preco', 'DY', 'PVP']].style
                .format({'Preco': 'R$ {:.2f}', 'DY': '{:.2f}%', 'PVP': '{:.2f}'})
                .background_gradient(subset=['DY'], cmap='Greens'),
                use_container_width=True
            )

    with tab3:
        st.subheader("Ranking Benjamin Graham")
        if 'Potencial_Graham' in df.columns:
            df_graham = df[df['Potencial_Graham'] > 0].sort_values(by='Potencial_Graham', ascending=False).head(30)
            st.dataframe(
                df_graham[['Ticker', 'Preco', 'Preco_Justo_Graham', 'Potencial_Graham', 'PL', 'PVP']].style
                .format({'Preco': 'R$ {:.2f}', 'Preco_Justo_Graham': 'R$ {:.2f}', 'Potencial_Graham': '{:.2f}%'})
                .bar(subset=['Potencial_Graham'], color='lightgreen'),
                use_container_width=True
            )

    with tab4:
        st.subheader("Ranking Magic Formula (Greenblatt)")
        if 'Score_Magic' in df.columns:
            # Remove quem não tem score
            df_magic_view = df.dropna(subset=['Score_Magic']).sort_values(by='Score_Magic', ascending=True).head(30)
            
            st.dataframe(
                df_magic_view[['Ticker', 'Preco', 'EV_EBIT', 'ROIC', 'Score_Magic']].style
                .format({'Preco': 'R$ {:.2f}', 'EV_EBIT': '{:.2f}', 'ROIC': '{:.2f}%', 'Score_Magic': '{:.0f}'})
                .background_gradient(subset=['Score_Magic'], cmap='Blues_r'),
                use_container_width=True
            )

else:
    st.warning("Aguardando dados... Se demorar muito, recarregue a página.")
