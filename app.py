import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from io import StringIO
import unicodedata

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(page_title="Titanium XVI | Deep Mind", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    
    /* Títulos e Métricas */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #e6edf3; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #3fb950; font-family: 'Roboto Mono', monospace; }
    
    /* Report Card - Estilo Relatório PDF */
    .report-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 25px;
        margin-bottom: 20px;
        font-family: 'Segoe UI', sans-serif;
    }
    .report-title { color: #58a6ff; font-size: 1.2rem; font-weight: bold; margin-bottom: 10px; border-bottom: 1px solid #30363d; padding-bottom: 5px; }
    .report-text { color: #c9d1d9; font-size: 0.95rem; line-height: 1.6; text-align: justify; }
    .highlight-good { color: #3fb950; font-weight: bold; }
    .highlight-bad { color: #f85149; font-weight: bold; }
    .highlight-neutral { color: #d2a8ff; font-weight: bold; }
    
    /* Tabelas */
    .stDataFrame { border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Titanium XVI: Deep Mind Analyst")

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
            'crescrec5a': 'Cresc_5a', 'liqcorr': 'LiqCorrente'
        }
        cols = [c for c in rename_map.keys() if c in df.columns]
        df = df[cols].rename(columns=rename_map)
        
        for col in df.columns:
            if col != 'Ticker' and df[col].dtype == object:
                df[col] = df[col].apply(clean_float)
                
        for col in ['DY', 'ROE', 'ROIC', 'MargemLiquida', 'MargemEbit', 'Cresc_5a']:
            if col in df.columns and df[col].mean() < 1: df[col] *= 100
            
        cols_req = ['PL', 'PVP', 'Preco', 'DY', 'EV_EBIT', 'ROIC', 'ROE', 'MargemLiquida', 'Div_Patrimonio', 'Cresc_5a']
        for c in cols_req: 
            if c not in df.columns: df[c] = 0.0

        def get_setor(t):
            t = t[:4]
            if t in ['ITUB','BBDC','BBAS','SANB']: return 'Financeiro'
            if t in ['VALE','CSNA','GGBR','USIM']: return 'Materiais'
            if t in ['PETR','PRIO','UGPA','CSAN']: return 'Petróleo'
            if t in ['MGLU','LREN','ARZZ','PETZ']: return 'Varejo'
            if t in ['WEGE','EMBR','TUPY','RAPT']: return 'Industrial'
            if t in ['TAEE','TRPL','ELET','CPLE']: return 'Elétricas'
            return 'Geral'
        df['Setor'] = df['Ticker'].apply(get_setor)
        
        # Rankings
        lpa = np.where(df['PL']!=0, df['Preco']/df['PL'], 0)
        vpa = np.where(df['PVP']!=0, df['Preco']/df['PVP'], 0)
        df['Graham_Fair'] = np.where((lpa>0)&(vpa>0), np.sqrt(22.5 * lpa * vpa), 0)
        df['Upside'] = np.where((df['Graham_Fair']>0), ((df['Graham_Fair']-df['Preco'])/df['Preco'])*100, -999)
        
        return df
    except: return pd.DataFrame()

# --- 3. CÉREBRO DEEP MIND (ANÁLISE COMPLEXA) ---
def deep_mind_analysis(ticker, row_fundamentus):
    """
    Realiza uma análise profunda cruzando dados do Fundamentus (Snapshot) 
    com dados Históricos do Yahoo (Tendência).
    """
    report = {
        "Executive_Summary": "",
        "Growth_Analysis": "",
        "Margin_Analysis": "",
        "Valuation_Analysis": "",
        "Risk_Analysis": ""
    }
    
    try:
        # Baixa dados contábeis
        stock = yf.Ticker(ticker + ".SA")
        fin_anual = stock.financials.T.sort_index(ascending=True) # Anual
        fin_quart = stock.quarterly_financials.T.sort_index(ascending=True) # Trimestral
        
        # 1. ANÁLISE DE CRESCIMENTO (Growth)
        growth_txt = []
        if len(fin_anual) >= 3 and 'Total Revenue' in fin_anual.columns:
            rev_last = fin_anual['Total Revenue'].iloc[-1]
            rev_3y = fin_anual['Total Revenue'].iloc[-3]
            cagr_rev = ((rev_last / rev_3y) ** (1/3) - 1) * 100
            
            growth_txt.append(f"A empresa apresenta um **CAGR de Receita (3 anos)** de <span class='{'highlight-good' if cagr_rev > 10 else 'highlight-neutral'}'>{cagr_rev:.1f}%</span>.")
            
            # Alavancagem Operacional (Lucro cresceu mais que receita?)
            if 'Operating Income' in fin_anual.columns:
                op_inc_last = fin_anual['Operating Income'].iloc[-1]
                op_inc_3y = fin_anual['Operating Income'].iloc[-3]
                try:
                    cagr_op = ((op_inc_last / op_inc_3y) ** (1/3) - 1) * 100
                    if cagr_op > cagr_rev + 2:
                        growth_txt.append("Destaca-se a <span class='highlight-good'>Alavancagem Operacional Positiva</span>: o lucro operacional cresceu mais rápido que a receita, indicando ganho de eficiência.")
                    elif cagr_op < cagr_rev - 5:
                        growth_txt.append("<span class='highlight-bad'>Alerta de Eficiência:</span> O lucro operacional não acompanhou o crescimento da receita, sugerindo aumento desproporcional de custos.")
                except: pass
        else:
            growth_txt.append("Dados históricos insuficientes para cálculo de CAGR preciso.")
            
        report["Growth_Analysis"] = " ".join(growth_txt)

        # 2. ANÁLISE DE MARGENS (Tendência)
        margin_txt = []
        margem_atual = row_fundamentus['MargemLiquida']
        
        # Tenta calcular média histórica
        if len(fin_anual) >= 3 and 'Net Income' in fin_anual.columns and 'Total Revenue' in fin_anual.columns:
            hist_margins = (fin_anual['Net Income'] / fin_anual['Total Revenue']) * 100
            avg_margin = hist_margins.mean()
            
            if margem_atual > avg_margin * 1.2:
                margin_txt.append(f"A Margem Líquida atual ({margem_atual:.1f}%) está <span class='highlight-good'>acima da média histórica</span> ({avg_margin:.1f}%), sugerindo um momento de pico de ciclo ou melhoria estrutural.")
            elif margem_atual < avg_margin * 0.8:
                margin_txt.append(f"A Margem Líquida atual ({margem_atual:.1f}%) está <span class='highlight-bad'>comprimida</span> em relação à média histórica ({avg_margin:.1f}%). Investigar pressão de custos.")
            else:
                margin_txt.append(f"As margens seguem estáveis e em linha com o histórico da empresa ({avg_margin:.1f}%).")
        
        report["Margin_Analysis"] = " ".join(margin_txt)

        # 3. VALUATION COMPARATIVO (Ben Graham + P/L Histórico)
        val_txt = []
        upside = row_fundamentus['Upside']
        pl_atual = row_fundamentus['PL']
        
        if upside > 30:
            val_txt.append(f"O modelo de Benjamin Graham aponta um <span class='highlight-good'>Upside Teórico de {upside:.0f}%</span>, sugerindo forte desconto patrimonial e de lucros.")
        elif upside < -20:
            val_txt.append(f"O modelo de Graham sugere que o ativo está <span class='highlight-bad'>negociado com prêmio</span> sobre seu valor intrínseco conservador.")
            
        if pl_atual < 5 and pl_atual > 0:
            val_txt.append("O múltiplo P/L abaixo de 5x classifica o ativo como **Deep Value**, embora exija caut
