import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO DE TERMINAL ELITE ---
st.set_page_config(page_title="Titanium XVII | Omni", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    
    /* Métricas e Títulos */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #e6edf3; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; color: #00ffbf; font-family: 'Roboto Mono', monospace; font-weight: 700; }
    
    /* Card de Relatório */
    .report-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .report-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 10px;
    }
    .text-bull { color: #3fb950; font-weight: 600; }
    .text-bear { color: #f85149; font-weight: 600; }
    .text-neutral { color: #8b949e; }
    
    /* Tabelas */
    .stDataFrame { border: 1px solid #30363d; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium XVII: Omni-Analyst")

# --- 2. MOTOR DE DADOS ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

def normalize_cols(cols):
    new = []
    for c in cols:
        n = unicodedata.normalize('NFKD', c)
        c = "".join([x for x in n if not unicodedata.combining(x)]).lower()
        c = c.replace('.', '').replace('/', '').replace(' ', '')
        new.append(c)
    return new

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        df.columns = normalize_cols(df.columns)
        
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'divyield': 'DY',
            'evebit': 'EV_EBIT', 'roic': 'ROIC', 'roe': 'ROE', 'liq2meses': 'Liquidez',
            'mrgliq': 'MargemLiquida', 'mrgebit': 'MargemEbit', 'divbrutpatr': 'Div_Patrimonio',
            'crescrec5a': 'Cresc_5a', 'liqcorr': 'LiqCorrente', 'psr': 'PSR'
        }
        
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        for col in df.columns:
            if col != 'Ticker': df[col] = df[col].apply(clean_float)
            
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        # Classificação Setorial
        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND']: return 'Construção'
            return 'Geral'
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings e Segurança
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        
        return df
    except: return pd.DataFrame()

# --- 3. CÉREBRO OMNI-ANALYST (IA DE TESE) ---
def omni_analysis(ticker, row_fundamentus):
    """
    Gera tese completa, score multidimensional e dados para radar.
    """
    report = {"Valuation": "", "Quality": "", "Growth": "", "Risk": "", "Score_Data": {}}
    
    # 1. VALUATION SCORE
    val_score = 0
    val_txt = []
    
    # P/L
    if row_fundamentus['PL'] <= 0:
        val_txt.append("🔴 Prejuízo recorrente (P/L negativo).")
        val_score = 0
    elif row_fundamentus['PL'] < 6:
        val_txt.append(
            "🟢 **Deep Value:** O múltiplo P/L abaixo de 6x indica forte desconto, "
            "mas exige verificação de recorrência de lucros."
        )
        val_score = 10
    elif row_fundamentus['PL'] < 15:
        val_txt.append("🔵 **Preço Justo:** Negociada em múltiplos razoáveis de mercado.")
        val_score = 7
    else:
        val_txt.append("🟡 **Prêmio de Crescimento:** P/L esticado, o mercado exige forte expansão futura.")
        val_score = 3
        
    # Graham
    if row_fundamentus['Upside'] > 30:
        val_txt.append(f"🟢 **Graham:** Margem de segurança de {row_fundamentus['Upside']:.0f}%.")
        val_score += 2
        
    report['Valuation'] = " ".join(val_txt)

    # 2. QUALITY SCORE (Eficiência)
    qual_score = 0
    qual_txt = []
    
    if row_fundamentus['ROE
