import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO DE TERMINAL ---
st.set_page_config(page_title="Titanium Pro XII", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0b0e11; }
    
    /* Métricas */
    [data-testid="stMetricValue"] { font-size: 1.3rem; color: #00ffbf; font-family: 'Roboto Mono', monospace; font-weight: 700; }
    
    /* Tabelas Densas */
    .stDataFrame { border: 1px solid #30363d; border-radius: 5px; }
    
    /* Card da IA */
    .ai-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ffbf;
        margin-bottom: 20px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro XII: AI Analyst Edition")

# --- 2. FUNÇÕES DE DADOS ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

def normalize_cols(cols):
    new_cols = []
    for col in cols:
        nfkd_form = unicodedata.normalize('NFKD', col)
        col = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        col = col.replace('.', '').replace('/', '').replace(' ', '').lower()
        new_cols.append(col)
    return new_cols

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers)
        dfs = pd.read_html(StringIO(r.text), decimal=',', thousands='.')
        if not dfs: return pd.DataFrame()
        df = dfs[0]
        
        # Normalização
        df.columns = normalize_cols(df.columns)
        
        # Mapeamento Completo
        rename_map = {
            'papel': 'Ticker', 'cotacao': 'Preco', 'pl': 'PL', 'pvp': 'PVP', 'psr': 'PSR',
            'divyield': 'DY', 'pativo': 'P_Ativo', 'pcapgiro': 'P_CapGiro',
            'pebit': 'P_EBIT', 'pativcircliq': 'P_AtivCircLiq',
            'evebit': 'EV_EBIT', 'evebitda': 'EV_EBITDA', 'mrgebit': 'MargemEbit',
            'mrgliq': 'MargemLiquida', 'liqcorr': 'LiqCorrente',
            'roic': 'ROIC', 'roe': 'ROE', 'liq2meses': 'Liquidez',
            'patrimliq': 'Patrimonio', 'divbrutpatr': 'Div_Patrimonio',
            'crescrec5a': 'Cresc_5a'
        }
        
        df.rename(columns=rename_map, inplace=True)
        
        # Limpeza
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        # Ajuste Percentual
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100

        # Garantia de Colunas
        req_cols = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a', 'PSR', 'LiqCorrente']
        for c in req_cols:
            if c not in df.columns: df[c] = 0.0

        # Classificação Setorial
        def get_setor(t):
            t = str(t)[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN','CMIN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE','NEOE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            return
