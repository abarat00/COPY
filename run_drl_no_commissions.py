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
output_dir = f'results/{ticker}_no_commissions'

# Crea directory di output se non esiste
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/weights', exist_ok=True)
os.makedirs(f'{output_dir}/test', exist_ok=True)
os.makedirs(f'{output_dir}/analysis', exist_ok=True)

# Carica il dataset
print(f"Caricamento dati per {ticker}...")
df = pd.read_csv(csv_path)

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

# Parametri per l'ambiente
max_steps = min(1000, len(df_train) - 10)  # Limita la lunghezza massima dell'episodio
print(f"Lunghezza massima episodio: {max_steps} timestep")

# Inizializza l'ambiente (senza commissioni)
print("Inizializzazione dell'ambiente (senza commissioni)...")
env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_train) - 1,
    lambd=0.05,             # Penalità per posizioni grandi ridotta
    psi=0.1,                # Penalità per costi di trading molto ridotta
    cost="trade_l1",
    max_pos=4.0,            # Posizione massima più grande
    squared_risk=False,
    penalty="tanh",
    alpha=2,
    beta=2,
    clip=True,
    scale_reward=5,
    df=df_train,
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=max_steps,
    # Parametri che eliminano le commissioni
    free_trades_per_month=10000,  # Praticamente infinito
    commission_rate=0.0,          # Commissione percentuale a zero
    min_commission=0.0,           # Commissione minima a zero
    # Nuovi parametri per behavior shaping
    trading_frequency_penalty_factor=0.1,  # Leggera penalità per trading frequente
    position_stability_bonus_factor=0.1    # Leggero bonus per stabilità
)

# Parametri di training
total_episodes = 200
save_freq = 10
learn_freq = 20

# Inizializza l'agente
print("Inizializzazione dell'agente DDPG...")
agent = Agent(
    memory_type="prioritized",
    batch_size=256,         # Batch size grande
    max_step=max_steps,
    theta=0.1,
    sigma=0.2               # Rumore moderato
)

# Parametri per l'addestramento
train_params = {
    'tau_actor': 0.01,
    'tau_critic': 0.05,
    'lr_actor': 1e-5,
    'lr_critic': 2e-4,
    'weight_decay_actor': 1e-6,
    'weight_decay_critic': 2e-5,
    'total_steps': 2000,
    'weights': f'{output_dir}/weights/',
    'freq': save_freq,
    'fc1_units_actor': 128,
    'fc2_units_actor': 64,
    'fc1_units_critic': 256,
    'fc2_units_critic': 128,
    'learn_freq': learn_freq,
    'decay_rate': 1e-6,
    'explore_stop': 0.1,
    'tensordir': f'{output_dir}/runs/',
    'progress': "tqdm",     # Mostra barra di avanzamento
}

# Avvia il training
print(f"Avvio del training per {ticker} - {total_episodes} episodi (senza commissioni)...")
agent.train(
    env=env,
    total_episodes=total_episodes,
    **train_params
)

print(f"Training completato per {ticker}!")
print(f"I modelli addestrati sono stati salvati in: {output_dir}/weights/")
print(f"I log per TensorBoard sono stati salvati in: {output_dir}/runs/")

# Valutazione sul dataset di test
print("\nAvvio della valutazione sul dataset di test...")

# Funzione per valutare un modello
def evaluate_model(model_file, env, agent):
    """Valuta le performance di un modello sul dataset di test."""
    model_path = os.path.join(f'{output_dir}/weights/', model_file)
    
    # Carica il modello
    from models import Actor
    model = Actor(env.state_size, fc1_units=128, fc2_units=64)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Assegna il modello all'agente
    agent.actor_local = model
    
    # Resetta l'ambiente all'inizio del test
    env.reset()
    state = env.get_state()
    done = env.done

    # Registra le azioni, posizioni e ricompense
    positions = [0]  # Inizia con posizione 0
    actions = []
    rewards = []
    
    # Esegui un singolo episodio attraverso tutti i dati di test
    while not done:
        action = agent.act(state, noise=False)  # Nessun rumore durante il test
        actions.append(action)
        reward = env.step(action)
        rewards.append(reward)
        state = env.get_state()
        positions.append(env.pi)
        done = env.done

    # Calcola metriche di performance
    cumulative_reward = np.sum(rewards)
    sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)  # Annualizzato
    cum_rewards = np.cumsum(rewards)
    running_max = np.maximum.accumulate(cum_rewards)
    drawdowns = cum_rewards - running_max
    max_drawdown = np.min(drawdowns)
    
    # Analisi del comportamento di trading
    n_trades = sum(1 for a in actions if abs(a) > 1e-6)
    avg_position = np.mean(positions)
    turnover = sum(abs(positions[i] - positions[i-1]) for i in range(1, len(positions)))

    return {
        'cumulative_reward': cumulative_reward,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'avg_position': avg_position,
        'turnover': turnover,
        'positions': positions,
        'actions': actions,
        'rewards': rewards
    }

# Inizializza l'ambiente di test
test_env = Environment(
    sigma=0.1,
    theta=0.1,
    T=len(df_test) - 1,
    lambd=0.05,
    psi=0.1,
    cost="trade_l1",
    max_pos=4.0,
    squared_risk=False,
    penalty="tanh",
    alpha=2,
    beta=2,
    clip=True,
    scale_reward=5,
    df=df_test,
    norm_params_path=norm_params_path,
    norm_columns=norm_columns,
    max_step=len(df_test),
    # Parametri senza commissioni anche per il test
    free_trades_per_month=10000,
    commission_rate=0.0,
    min_commission=0.0,
    trading_frequency_penalty_factor=0.1,
    position_stability_bonus_factor=0.1
)

# Ottieni la lista dei modelli salvati
weights_dir = f'{output_dir}/weights'
model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
model_files.sort(key=lambda x: int(x[4:-4]) if x[4:-4].isdigit() else 0)

if not model_files:
    print("Nessun modello trovato per la valutazione.")
else:
    # Scegli modelli da valutare (primo, medio, ultimo)
    if len(model_files) > 3:
        selected_models = [
            model_files[0],  # primo modello
            model_files[len(model_files) // 2],  # modello a metà
            model_files[-1]  # ultimo modello
        ]
    else:
        selected_models = model_files
    
    # Valuta i modelli selezionati
    evaluation_results = []
    
    for model_file in selected_models:
        print(f"Valutazione del modello {model_file}...")
        results = evaluate_model(model_file, test_env, agent)
        results['model'] = model_file
        evaluation_results.append(results)
        
        print(f"  Ricompensa cumulativa: {results['cumulative_reward']:.2f}")
        print(f"  Sharpe ratio: {results['sharpe']:.2f}")
        print(f"  Max drawdown: {results['max_drawdown']:.2f}")
        print(f"  Numero di trade: {results['n_trades']}")
        print(f"  Posizione media: {results['avg_position']:.2f}")
        print(f"  Turnover totale: {results['turnover']:.2f}")
    
    # Salva i risultati
    if evaluation_results:
        # Converti i risultati in DataFrame
        eval_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in evaluation_results])
        eval_df.to_csv(f"{output_dir}/test/evaluation_results.csv", index=False)
        print(f"Risultati della valutazione salvati in: {output_dir}/test/evaluation_results.csv")
        
        # Trova il miglior modello
        best_model_idx = np.argmax([r['cumulative_reward'] for r in evaluation_results])
        best_model = evaluation_results[best_model_idx]
        
        print(f"\nMiglior modello: {best_model['model']}")
        print(f"  Ricompensa cumulativa: {best_model['cumulative_reward']:.2f}")
        print(f"  Sharpe ratio: {best_model['sharpe']:.2f}")
        print(f"  Max drawdown: {best_model['max_drawdown']:.2f}")
        
        # Crea un grafico del miglior modello
        plt.figure(figsize=(14, 10))
        
        # Plot delle posizioni
        plt.subplot(3, 1, 1)
        plt.plot(best_model['positions'][:-1], label='Posizione', color='blue')
        plt.title(f'Posizioni del modello {best_model["model"]}')
        plt.ylabel('Posizione')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot delle azioni
        plt.subplot(3, 1, 2)
        plt.plot(best_model['actions'], label='Azioni', color='red')
        plt.title('Azioni (trades)')
        plt.ylabel('Azioni')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot della ricompensa cumulativa
        plt.subplot(3, 1, 3)
        plt.plot(np.cumsum(best_model['rewards']), label='Ricompensa cumulativa', color='green')
        plt.title('Ricompensa cumulativa')
        plt.xlabel('Timestep')
        plt.ylabel('Ricompensa cumulativa')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test/best_model_performance.png")
        print(f"Grafico delle performance salvato in: {output_dir}/test/best_model_performance.png")
    else:
        print("Nessun risultato di valutazione disponibile.")