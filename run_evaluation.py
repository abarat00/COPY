from evaluation import test_models, plot_bars, plot_hist
from env import Environment
import matplotlib.pyplot as plt

def main():
    # 1) Istanzia l'ambiente (occhio ai parametri necessari per il tuo costruttore)
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
        # e così via...
    )

    # 2) Richiama test_models
    scores, scores_episodes, scores_cumsum, pnls, positions = test_models(
        path_weights="weights/",  # cartella in cui hai i file ddpg_0.pth, ddpg_1.pth, ...
        env=env,
        fc1_units=16,
        fc2_units=8,
        random_state=42,
        n_episodes=10
    )

    # 3) Visualizza i risultati
    print("Scores (media punteggi) per ogni modello:", scores)

    # 4) Genera alcuni plot di esempio
    plot_bars(scores)
    plot_hist(model_key=0, scores_episodes=scores_episodes)  # es. key=0

    # Se vuoi mostrare i grafici in una finestra, su alcune configurazioni
    # potresti dover aggiungere:
    # plt.show()

if __name__ == "__main__":
    main()
