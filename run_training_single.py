import os
import torch
import numpy as np
from agent import Agent
from env import Environment
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configurazione
ticker = "ARKG"  # Ticker da utilizzare
norm_params_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/json/{ticker}_norm_params.json'
csv_path = f'/Users/Alessandro/Desktop/DRL/NAS Results/Multi_Ticker/Normalized_RL_INPUT/{ticker}/{ticker}_normalized.csv'
output_dir = f'results/{ticker}'

# Crea directory di output se non esiste
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/weights', exist_ok=True)
os.makedirs(f'{output_dir}/test', exist_ok=True)  # Aggiungi cartella per i dati di test

# Verifica esistenza dei file necessari
if not os.path.exists(norm_params_path):
    print("File dei parametri di normalizzazione non trovato. Esecuzione create_norm_params.py...")
    os.system("python3 create_norm_params.py")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File CSV dei dati normalizzati non trovato: {csv_path}")

# Definizione delle feature da utilizzare 
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
    "pred_lstm", "pred_gru", "pred_blstm", "pred_lstm_direction",
    "pred_gru_direction", "pred_blstm_direction"
]

# Carica il dataset
print(f"Caricamento dati per {ticker}...")
df = pd.read_csv(csv_path)

# Verifica la presenza di tutte le colonne necessarie
missing_cols = [col for col in norm_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonne mancanti nel dataset: {missing_cols}")

# Ordina il dataset per data (se presente)
if 'date' in df.columns:
    print("Ordinamento del dataset per data...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    print(f"Intervallo temporale: {df['date'].min()} - {df['date'].max()}")

# Stampa info sul dataset
print(f"Dataset caricato: {len(df)} righe x {len(df.columns)} colonne")

# Separazione in training e test
train_size = int(len(df) * 0.8)  # 80% per training, 20% per test
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

print(f"Divisione dataset: {len(df_train)} righe per training, {len(df_test)} righe per test")

if 'date' in df.columns:
    print(f"Periodo di training: {df_train['date'].min()} - {df_train['date'].max()}")
    print(f"Periodo di test: {df_test['date'].min()} - {df_test['date'].max()}")

# Salva il dataset di test per usi futuri
test_dir = f'{output_dir}/test'
os.makedirs(test_dir, exist_ok=True)
df_test.to_csv(f'{test_dir}/{ticker}_test.csv', index=False)
print(f"Dataset di test salvato in: {test_dir}/{ticker}_test.csv")

# Parametri per l'ambiente
max_steps = min(1000, len(df_train) - 10)  # Limita la lunghezza massima dell'episodio
print(f"Lunghezza massima episodio: {max_steps} timestep")

# Inizializza l'ambiente
env = Environment(
    sigma=0.1,            # Per compatibilità con il vecchio codice
    theta=0.1,            # Per compatibilità con il vecchio codice
    T=len(df_train) - 1,  # Usa lunghezza del dataset di training
    lambd=0.3,            # Peso penalità posizione
    psi=0.5,              # Peso costi di trading
    cost="trade_l1",      # Modello di costo
    max_pos=2,            # Posizione massima
    squared_risk=False,   # Usa rischio quadratico
    penalty="tanh",       # Tipo di penalità
    alpha=10,             # Parametro penalità
    beta=10,              # Parametro penalità
    clip=True,            # Limita le posizioni
    scale_reward=10,      # Scala per le ricompense
    df=df_train,          # Usa il dataset di training
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=max_steps    # Lunghezza massima episodio
)

# Parametri di training
total_episodes = 100      # Numero di episodi
learn_freq = 50           # Frequenza di apprendimento
save_freq = 10            # Frequenza di salvataggio dei modelli

# Inizializza l'agente
print("Inizializzazione dell'agente DDPG...")
agent = Agent(
    memory_type="prioritized",
    batch_size=64,
    max_step=max_steps,
    theta=0.1,            # Parametro rumore OU
    sigma=0.1             # Parametro rumore OU
)

# Avvia il training
print(f"Avvio del training per {ticker} - {total_episodes} episodi...")
agent.train(
    env=env,
    total_episodes=total_episodes,
    tau_actor=0.1,
    tau_critic=0.01,
    lr_actor=1e-4,
    lr_critic=1e-3,
    weight_decay_actor=0,
    weight_decay_critic=1e-4,
    total_steps=1000,
    weights=f'{output_dir}/weights/',
    freq=save_freq,
    fc1_units_actor=128,
    fc2_units_actor=64,
    fc1_units_critic=256,
    fc2_units_critic=128,
    learn_freq=learn_freq,
    decay_rate=1e-5,
    tensordir=f'{output_dir}/runs/',
    progress="tqdm",  # Mostra barra di avanzamento
)

print(f"Training completato per {ticker}!")
print(f"I modelli addestrati sono stati salvati in: {output_dir}/weights/")
print(f"I log per TensorBoard sono stati salvati in: {output_dir}/runs/")
