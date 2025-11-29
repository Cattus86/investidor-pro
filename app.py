import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO

# --- 1. CONFIGURAÇÃO VISUAL DE TERMINAL ---
st.set_page_config(page_title="Titanium Pro VI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    
    /* Métricas estilo Neon */
    [data-testid="stMetricValue"] { font-size: 1.4rem; color: #00ffbf; font-family: 'Roboto Mono', monospace; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
    
    /* Tabela Profissional Compacta */
    .stDataFrame { font-size: 12px; }
    div[data-testid="stDataFrame"] div[class*="stDataFrame"] { border: 1px solid #333; }
    
    /* Abas */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #161b22; padding: 5px; }
    .stTabs [data-baseweb="tab"] { height: 30px; font-size: 12px; color: #ccc; border: none; }
    .stTabs [aria-selected="true"] { background-color: #238636 !important; color: white !important; }
    
    /* Expander dos Filtros */
    .streamlit-expanderHeader { background-color: #161b22; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium Pro VI: Full Terminal")

# --- 2. MOTOR DE DADOS FUNDAMENTALISTA (TODAS AS COLUNAS) ---
def clean_float(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.').replace('%', '').strip()
        try: return float(val)
        except: return 0.0
    return float(val) if val else 0.0

@st.cache_data(ttl=300, show_spinner=False)
def get_full_market_data():
    url = 'https://www.fundamentus.com.br/resultado.php'
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        r = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(r.text), decimal=',', thousands='.')[0]
        
        # MAPEAMENTO TOTAL (20+ Indicadores)
        rename_map = {
            'Papel': 'Ticker', 'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'PSR': 'PSR',
            'Div.Yield': 'DY', 'P/Ativo': 'P_Ativo', 'P/Cap.Giro': 'P_CapGiro',
            'P/EBIT': 'P_EBIT', 'P/Ativ Circ Liq': 'P_AtivCircLiq',
            'EV/EBIT': 'EV_EBIT', 'EV/EBITDA': 'EV_EBITDA', 'Mrg Ebit': 'MargemEbit',
            'Mrg. Líq.': 'MargemLiquida', 'Liq. Corr.': 'LiqCorrente',
            'ROIC': 'ROIC', 'ROE': 'ROE', 'Liq.2meses': 'Liquidez',
            'Patrim. Líq': 'Patrimonio', 'Dív.Brut/ Patr.': 'Div_Patrimonio',
            'Cresc. Rec.5a': 'Cresc_5a'
        }
        
        # Filtra e renomeia
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        # Limpeza Numérica Total
        for col in df.columns:
            if col != 'Ticker':
                if df[col].dtype == object: df[col] = df[col].apply(clean_float)
        
        # Ajuste de Escala Percentual (Para ficar legível: 10.0 ao invés de 0.10)
        pct_cols = ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']
        for col in pct_cols:
            if col in df.columns:
                # Heurística: se a média for muito baixa (<1), multiplica por 100
                if df[col].mean() < 1: df[col] *= 100

        # Classificação Setorial
        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB','BPAC','B3SA']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM','SUZB','KLBN']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN','RRRP','VBBR']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ','AMER','SOMA']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT','POMO']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE','EQTL','CMIG','EGIE']: return 'Elétricas'
            if t in ['RADL','RDOR','HAPV','FLRY','QUAL']: return 'Saúde'
            if t in ['CYRE','EZTC','MRVE','TEND','JHSF']: return 'Construção'
            if t in ['VIVT','TIMS','LWSA','TOTS']: return 'Tecnologia'
            return 'Geral'
        
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings Proprietários
        df['Graham'] = np.where((df['PL']>0)&(df['PVP']>0), np.sqrt(22.5 * df['PL'] * df['PVP']), 0) # Graham Number simplificado
        # Fórmula de Graham Preço Justo = Raiz(22.5 * LPA * VPA). 
        # Adaptando com dados da tabela: Justo = Raiz(22.5 * (P/PL) * (P/PVP) ) -> Não, isso dá erro.
        # Correção: Graham = Raiz(22.5 * LPA * VPA). 
        # Como não temos LPA e VPA diretos, calculamos: LPA = Preco/PL, VPA = Preco/PVP
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        
        df['Bazin_Fair'] = np.where(df['DY']>0, df['Preco'] * (df['DY']/6), 0) # Ajuste Bazin: Se paga 6%, preço é justo.
        
        # Magic Formula Score (Menor é melhor)
        df_m = df[(df['EV_EBIT']>0)&(df['ROIC']>0)].copy()
        if not df_m.empty:
            df_m['Score_Magic'] = df_m['EV_EBIT'].rank(ascending=True) + df_m['ROIC'].rank(ascending=False)
            df = df.merge(df_m[['Ticker', 'Score_Magic']], on='Ticker', how='left')
        else:
            df['Score_Magic'] = 99999

        return df
    except Exception as e:
        st.error(f"Erro Crítico de Dados: {e}")
        return pd.DataFrame()

# --- 3. MOTOR CONTÁBIL (ANÁLISE VERTICAL/HORIZONTAL) ---
def get_accounting_analysis(ticker):
    """Baixa Balanço e DRE e calcula AV e AH"""
    try:
        stock = yf.Ticker(ticker+".SA")
        inc = stock.financials.T.sort_index(ascending=True)
        if inc.empty: return None, None
        
        # Análise Vertical (Base: Total Revenue)
        if 'Total Revenue' in inc.columns:
            av = pd.DataFrame()
            receita = inc['Total Revenue']
            av['Receita Líquida'] = receita
            
            campos = {
                'Cost Of Revenue': 'CPV',
                'Gross Profit': 'Lucro Bruto',
                'Operating Income': 'EBIT',
                'Net Income': 'Lucro Líquido'
            }
            
            for en, pt in campos.items():
                if en in inc.columns:
                    av[pt] = inc[en]
                    av[f'{pt} AV%'] = (inc[en] / receita) * 100
            
            # Análise Horizontal (Crescimento Ano a Ano)
            ah = inc[list(campos.keys()) + ['Total Revenue']].pct_change() * 100
            ah.columns = [f"{c} Cresc. %" for c in ah.columns]
            
            return av.iloc[-4:], ah.iloc[-4:]
            
    except: return None, None
    return None, None

# --- 4. MOTOR MOMENTUM (YAHOO) ---
@st.cache_data(ttl=1800)
def get_momentum_data(tickers):
    """Baixa histórico para calcular Momentum"""
    if not tickers: return {}
    ts = [t+".SA" for t in tickers]
    try:
        h =
