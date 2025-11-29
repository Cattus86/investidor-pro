import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Investidor Pro | Ultimate", layout="wide", initial_sidebar_state="expanded")

# CSS para visual profissional (Dark Theme refinado)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 1.2rem; color: #00e676; }
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; white-space: pre-wrap; background-color: #1f2937; color: #aaa; border-radius: 4px; font-size: 14px;
    }
    .stTabs [aria-selected="true"] { background-color: #00e676 !important; color: black !important; font-weight: bold; }
    hr { margin-top: 0.5rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Investidor Pro: Ultimate Edition")
st.markdown("##### Terminal de Análise Quantitativa Completa")

# --- 2. MAPAS E CONSTANTES ---
MAPA_SETORES = {
    'Bancos': ['BBAS3', 'ITUB4', 'BBDC4', 'SANB11', 'BPAC11', 'ABCB4', 'BRSR6', 'ITSA4', 'BBSE3', 'CXSE3'],
    'Energia': ['PETR4', 'PETR3', 'PRIO3', 'VBBR3', 'UGPA3', 'CSAN3', 'ENAT3', 'RRRP3', 'RECV3', 'BRAV3'],
    'Elétricas': ['ELET3', 'ELET6', 'EGIE3', 'TRPL4', 'TAEE11', 'CPLE6', 'CMIG4', 'EQTL3', 'LIGT3', 'NEOE3', 'ALUP11', 'AURE3'],
    'Mineração/Sid': ['VALE3', 'CSNA3', 'GGBR4', 'GOAU4', 'USIM5', 'CMIN3', 'FESA4', 'CBAV3'],
    'Varejo': ['MGLU3', 'LREN3', 'ARZZ3', 'SOMA3', 'PETZ3', 'RDOR3', 'RADL3', 'AMER3', 'BHIA3', 'CEAB3', 'GUAR3'],
    'Indústria/Bens': ['WEGE3', 'EMBR3', 'TUPY3', 'RAPT4', 'POMO4', 'SHUL4', 'KEPL3'],
    'Construção': ['CYRE3', 'EZTC3', 'MRVE3', 'TEND3', 'JHSF3', 'DIRR3', 'CURY3'],
    'Saneamento': ['SBSP3', 'CSMG3', 'SAPR11', 'SAPR4', 'AMBP3'],
    'Agronegócio': ['SLCE3', 'AGRO3', 'SOJA3', 'TTEN3', 'SMTO3'],
    'Telecom/Tech': ['VIVT3', 'TIMS3', 'LWSA3', 'INTB3', 'TOTS3'],
    'Logística': ['STBP3', 'HBSA3', 'RAIL3']
}

def obter_setor(ticker):
    t = ticker.upper().strip()
    for s, l in MAPA_SETORES.items():
        if t in l: return s
    return "Outros"

def limpar_coluna(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

# --- 3. MOTOR DE DADOS ---
@st.cache_data(ttl=900, show_spinner=False)
def baixar_fundamentus():
    try:
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Mapeamento EXTENSIVO (Todos os indicadores relevantes)
        mapa = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'PSR': 'PSR',
            'Div.Yield': 'DY', 'P/Ativo': 'P_Ativo', 'P/Cap.Giro': 'P_CapGiro',
            'P/EBIT': 'P_EBIT', 'P/Ativ Circ Liq': 'P_AtivCircLiq',
            'EV/EBIT': 'EV_EBIT', 'EV/EBITDA': 'EV_EBITDA',
            'Mrg Ebit': 'MargemEbit', 'Mrg. Líq.': 'MargemLiquida',
            'Liq. Corr.': 'LiqCorrente', 'ROIC': 'ROIC', 'ROE': 'ROE',
            'Liq.2meses': 'Liquidez', 'Patrim. Líq': 'Patrimonio',
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        # Seleciona e renomeia
        cols_existentes = ['Ticker'] + [c for c in mapa.keys() if c in df.columns]
        df = df[cols_existentes].copy()
        df.rename(columns=mapa, inplace=True)
        
        # Limpeza Numérica
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(limpar_coluna)
            
        # Ajuste Percentual
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        df['Setor'] = df['Ticker'].apply(obter_setor)
        return df
    except Exception as e:
        st.error(f"Erro Fundamentus: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def calcular_metricas_tecnicas(df_base):
    # Otimização: Baixar dados técnicos apenas das Top 150 mais líquidas para não travar
    # (Usuários geralmente não querem ver gráfico de mico sem liquidez)
    df_top = df_base.nlargest(150, 'Liquidez').copy()
    tickers = [t + ".SA" for t in df_top['Ticker'].tolist()]
    
    if not tickers: return df_base

    try:
        # Baixa 1 ano para calcular Volatilidade (StdDev) e Momentum
        dados = yf.download(tickers, period="1y", progress=False)['Adj Close']
        
        if isinstance(dados, pd.Series): dados = dados.to_frame() # Correção bug 1 ticker
        
        res = {}
        for t_full in dados.columns:
            t_clean = t_full.replace('.SA', '')
            serie = dados[t_full].dropna()
            
            if len(serie) > 20:
                # 1. Momentum (6 Meses)
                try:
                    p_atual = serie.iloc[-1]
                    # Pega preço de ~126 dias úteis atrás (6 meses)
                    idx_6m = max(0, len(serie) - 126)
                    p_antigo = serie.iloc[idx_6m]
                    mom = ((p_atual - p_antigo) / p_antigo) * 100
                except: mom = 0
                
                # 2. Volatilidade Anualizada
                try:
                    retornos = serie.pct_change().dropna()
                    vol = retornos.std() * np.sqrt(252) * 100 # Anualizado em %
                except: vol = 0
                
                res[t_clean] = {'Momentum': mom, 'Volatilidade': vol}
        
        # Incorpora ao DataFrame
        df_base['Momentum'] = df_base['Ticker'].map(lambda x: res.get(x, {}).get('Momentum', 0))
        df_base['Volatilidade'] = df_base['Ticker'].map(lambda x: res.get(x, {}).get('Volatilidade', 0))
        
    except:
        df_base['Momentum'] = 0.0
        df_base['Volatilidade'] = 0.0
        
    return df_base

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

# --- 4. CÉREBRO DA ANÁLISE ---
def analisar_ativo_ia(row):
    score = 5
    txt = []
    
    # Valuation
    if row['PL'] < 5 and row['PL'] > 0: txt.append("🟢 **Valuation:** Muito barato (P/L < 5)."); score += 2
    elif row['PL'] > 30: txt.append("🔴 **Valuation:** Caro (P/L > 30)."); score -= 2
    
    if row['PVP'] < 0.8 and row['PVP'] > 0: txt.append
