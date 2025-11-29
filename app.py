import streamlit as st
import pandas as pd
import fundamentus
import numpy as np
import plotly.express as px

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Investidor Pro | Titanium", layout="wide")
st.title("💎 Investidor Pro: Titanium Edition")
st.markdown("### A Plataforma Definitiva: Sem erros, Análise Automática e Rankings.")

# --- 2. FUNÇÕES DE LIMPEZA E DADOS ---
def limpar_numero_ptbr(valor):
    """Converte números brasileiros (texto) para float do Python."""
    if isinstance(valor, str):
        # Remove pontos de milhar, troca vírgula por ponto e tira %
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
        
        # Mapeamento simplificado para evitar erros de sintaxe
        df.rename(columns={
            'Cotação': 'Preco', 
            'P/L': 'PL', 
            'P/VP': 'PVP', 
            'Div.Yield': 'DY',
            'ROE': 'ROE', 
            'ROIC': 'ROIC', 
            'EV/EBIT': 'EV_EBIT',
            'Liq.2meses': 'Liquidez', 
            'Mrg. Líq.': 'MargemLiquida',
            'Dív.Brut/ Patr.': 'Div_Patrimonio', 
            'Cresc. Rec.5a': 'Cresc_5a'
        }, inplace=True)
        
        # Limpeza Numérica em todas as colunas exceto Ticker
        for col in df.columns:
            if col != 'Ticker':
                df[col] = df[col].apply(limpar_numero_ptbr)
        
        # Ajustes de Escala (se vier 0.10 transformar em 10.0)
        # Verifica se precisa multiplicar por 100 baseado na média
        if 'DY' in df.columns and df['DY'].mean() < 1: df['DY'] *= 100
        if 'ROE' in df.columns and df['ROE'].mean() < 1: df['ROE'] *= 100
        if 'MargemLiquida' in df.columns and df['MargemLiquida'].mean() < 1: df['MargemLiquida'] *= 100
        
        return df
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    # Graham
    # Evita divisão por zero
    df['LPA'] = np.where(df['PL'] != 0, df['Preco'] / df['PL'], 0)
    df['VPA'] = np.where(df['PVP'] != 0, df['Preco'] / df['PVP'], 0)
    
    mask_valida = (df['LPA'] > 0) & (df['VPA'] > 0)
    df.loc[mask_valida, 'Graham_Valor'] = np.sqrt(22.5 * df.loc[mask_valida, 'LPA'] * df.loc[mask_valida, 'VPA'])
    df['Graham_Valor'] = df['Graham_Valor'].fillna(0)
    
    # Potencial (Upside)
    df['Graham_Upside'] = np.where(
        (df['Graham_Valor'] > 0) & (df['Preco'] > 0),
        ((df['Graham_Valor'] - df['Preco']) / df['Pre
