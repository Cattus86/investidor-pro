import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Titanium", layout="wide", initial_sidebar_state="expanded")
st.title("💎 Investidor Pro: Titanium Edition")

# --- 2. FUNÇÕES DE SUPORTE (LIMPEZA DE DADOS) ---
def limpar_numero_ptbr(valor):
    """Converte números brasileiros (texto) para float do Python."""
    if isinstance(valor, str):
        valor_limpo = valor.replace('.', '').replace(',', '.').replace('%', '').strip()
        try:
            return float(valor_limpo)
        except:
            return 0.0
    return float(valor) if valor else 0.0

@st.cache_data(ttl=300)
def carregar_dados_titanium():
    try:
        # Baixa dados brutos
        df = fundamentus.get_resultado_raw().reset_index()
        df.rename(columns={'papel': 'Ticker'}, inplace=True)
        
        # Mapa de renomeação de colunas
        mapa_colunas = {
            'Cotação': 'Preco', 'P/L': 'PL', 'P/VP': 'PVP', 'Div.Yield': 'DY',
            'ROE': 'ROE', 'ROIC': 'ROIC', 'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 'Mrg. Líq.': 'Margem
