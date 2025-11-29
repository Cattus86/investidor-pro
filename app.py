import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Investidor Pro | Diamond", layout="wide")
st.title("💎 Investidor Pro: Diamond Edition")

# --- ROBÔ ANALISTA (SIMULANDO O GEMINI) ---
def gerar_analise_automatica(row):
    """
    Esta função age como um analista, escrevendo um texto baseada nos dados.
    """
    ticker = row['Ticker']
    dy = row['DY']
    pl = row['PL']
    roe = row['ROE']
    pvp = row['PVP']
    divida = row['Div_Patrimonio']
    
    analise = []
    score = 0
    
    analise.append(f"### 🤖 Análise Rápida: {ticker}")
    
    # Avaliação de Preço
    if pl < 0:
        analise.append(f"🔴 **Cuidado:** A empresa está com prejuízo atualmente (P/L negativo).")
        score -= 2
    elif pl < 5:
        analise.append(f"🟢 **Preço:** A empresa está extremamente barata (P/L abaixo de 5). Pode ser uma oportunidade de ouro ou uma 'armadilha de valor'.")
        score += 2
    elif pl < 15:
        analise.append(f"🔵 **Preço:** A cotação parece justa em relação ao lucro.")
        score += 1
    else:
        analise.append(f"🟡 **Preço:** O mercado está pagando caro pelo lucro dessa empresa (P/L alto). Espera-se alto crescimento.")

    # Avaliação de Dividendos
    if dy > 12:
        analise.append(f"🟢 **Renda:** Dividend Yield excelente ({dy:.1f}%), mas verifique se é recorrente.")
        score += 2
    elif dy > 6:
        analise.append(f"🔵 **Renda:** Paga bons dividendos ({dy:.1f}%), acima da média do mercado.")
        score += 1
    else:
        analise.append(f"⚪ **Renda:** Não é o foco principal deste ativo no momento (DY baixo).")

    # Avaliação de Qualidade (ROE)
    if roe > 15:
        analise.append(f"🔥 **Eficiência:** A empresa é uma máquina de fazer dinheiro! ROE alto de {roe:.1f}%.")
        score += 2
    elif roe < 5:
        analise.append(f"❄️ **Eficiência:** A rentabilidade sobre o patrimônio está baixa ({roe:.1f}%).")
        score -= 1

    # Veredito Final
    analise.append("---")
    if score >= 4:
        analise.append(f"**Veredito:** ⭐⭐⭐⭐⭐ (Ativo de Alta Qualidade/Oportunidade)")
    elif score >= 2:
        analise.append(f"**Veredito:** ⭐⭐⭐ (Ativo Sólido/Observar)")
    else:
        analise.append(f"**Veredito:** ⭐ (Requer Cautela Extrema)")
        
    return "\n\n".join(analise)

# --- TRATAMENTO DE DADOS ---
def limpar_numero_ptbr(valor):
    if isinstance(valor, str):
        valor_limpo = valor.replace('.', '').replace(',', '.').replace('%', '').strip()
        try:
            return float(valor_limpo)
        except:
            return 0.0
    return float(valor) if valor else 0.0

@st.cache_data(ttl=300)
def carregar_dados_diamond():
    try:
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        mapa_colunas = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        colunas_uteis = [c for c in mapa_colunas.keys() if c in df.columns]
        df = df[['Ticker'] + colunas_uteis].copy()
        df.rename(columns=mapa_colunas, inplace=True)
        
        cols_numericas = ['Preco', 'PL', 'PVP', 'DY', 'ROE', 'ROIC', 'EV_EBIT', 'Liquidez', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a']
        for col in cols_numericas:
            if col in df.columns:
                df[col] = df[col].apply(limpar_numero_ptbr)
            else:
                df[col] = 0.0
        
        if 'DY' in df.columns and df['DY'].mean() < 1: df['DY'] *= 100
        if 'ROE' in df.columns and df['ROE'].mean() < 1: df['ROE'] *= 100
        if 'MargemLiquida' in df.columns and df['MargemLiquida'].mean() < 1: df['MargemLiquida'] *= 100
        
        return df
    except Exception as e:
        st.error(f"Erro: {e}")
        return pd.DataFrame()

def calcular_kpis(df):
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

# --- INTERFACE ---
with st.spinner('Carregando Big Data...'):
    df_raw = carregar_dados_diamond()

if not df_raw.empty:
    df = calcular_kpis(df_raw)
    
    # --- ÁREA DE ANÁLISE (Topo) ---
    st.markdown("### 🤖 Analista Virtual")
    st.info("💡 **Instrução:** Selecione uma ação na tabela abaixo (clique na caixa à esquerda) para ver a análise automática.")
    
    col_analise, col_filtros = st.columns([2, 1])
    
    # --- SIDEBAR/FILTROS (Agora na direita para ficar perto da tabela) ---
    with col_filtros:
        st.markdown("#### 🛠️ Filtros")
        liq_min = st.select_slider("Liquidez Mínima", options=[0, 1000, 50000, 200000, 1000000], value=50000)
        
        # NOVOS FILTROS
        somente_lucro = st.checkbox("Somente Empresas com Lucro", value=True)
        preco_min, preco_max = st.slider("Faixa de Preço (R$)", 0.0, 200.0, (1.0, 100.0))
        
        # Aplicação dos Filtros
        df_view = df[
            (df['Liquidez'] >= liq_min) &
            (df['Preco'] >= preco_min) &
            (df['Preco'] <= preco_max)
        ].copy()
        
        if somente_lucro:
            df_view = df_view[df_view['PL'] > 0]

    # --- TABELA INTERATIVA (COM SELEÇÃO) ---
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Dividendos", "💎 Graham", "✨ Magic Formula", "📈 Gráficos"])
    
    # Configuração comum das colunas
    cfg_padrao = {
        "Preco": st.column_config.NumberColumn("Preço", format="R$ %.2f"),
        "DY": st.column_config.ProgressColumn("Yield", format="%.2f%%", min_value=0, max_value=15),
        "PVP": st.column_config.NumberColumn
