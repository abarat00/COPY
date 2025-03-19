import os
import torch
from agent import Agent
from env import Environment
import pandas as pd

# Percorso al file JSON con i parametri di normalizzazione
ticker = "ARKG"
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'

# Se il file non esiste, esegui il file create_norm_params.py per crearlo
if not os.path.exists(norm_params_path):
    os.system("python3 create_norm_params.py")

# Definisci l'ordine esatto delle feature (le 64 feature)
norm_columns = [
    "open", "volume", "change", "day", "week", "adjCloseGold", "adjCloseSpy",
    "Credit_Spread", "Log_Close", "m_plus", "m_minus", "drawdown", "drawup",
    "s_plus", "s_minus", "upper_bound", "lower_bound", "avg_duration", "avg_depth",
    "cdar_95", "VIX_Close", "MACD", "MACD_Signal", "MACD_Histogram", "SMA5",
    "SMA10", "SMA15", "SMA20", "SMA25", "SMA30", "SMA36", "RSI5", "RSI14", "RSI20",
    "RSI25", "ADX5", "ADX10", "ADX15", "ADX20", "ADX25", "ADX30", "ADX35",
    "BollingerLower", "BollingerUpper", "WR5", "WR14", "WR20", "WR25",
    "SMA5_SMA20", "SMA5_SMA36", "SMA20_SMA36", "SMA5_Above_SMA20",
    "Golden_Cross", "Death_Cross", "BB_Position", "BB_Width",
    "BB_Upper_Distance", "BB_Lower_Distance", "Volume_SMA20", "Volume_Change_Pct",
    "Volume_1d_Change_Pct", "Volume_Spike", "Volume_Collapse", "GARCH_Vol",
    "pred_lstm", "pred_gru", "pred_blstm",
    "pred_lstm_direction", "pred_gru_direction", "pred_blstm_direction"
]

# Inizializza l'ambiente: si assume che i dati in ingresso siano già normalizzati.
env = Environment(
    sigma=0.1,
    theta=0.1,
    T=5000,
    lambd=0.3,
    psi=4,
    cost="trade_l1",
    max_pos=2,
    squared_risk=False,
    penalty="tanh",
    norm_params_path=norm_params_path,
    norm_columns=norm_columns
)

# Aggiorna l'ambiente con i dati normalizzati per il ticker scelto.
# Ad esempio, per il ticker "ARKG":
csv_path = '/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/ARKG/ARKG_normalized.csv'
df = pd.read_csv(csv_path)
# Supponiamo di iniziare dal primo timestep:
current_index = 0
env.update_raw_state(df, current_index)

# Inizializza l'agente (utilizziamo la memoria prioritizzata)
agent = Agent(memory_type="prioritized")

# Avvia un training di prova per 10 episodi (modifica learn_freq, freq e total_episodes se necessario)
agent.train(env, total_episodes=10, learn_freq=50, freq=5)

print("Training di prova per il ticker ARKG completato!")
