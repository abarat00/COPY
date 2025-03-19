import numpy as np
import json
from utils import build_ou_process

class Environment:
    """
    L'ambiente rappresenta il problema di ottimizzazione del portafoglio.
    Lo stato è costituito dalle feature già normalizzate (disponibili in self.raw_state)
    riordinate secondo l'ordine definito in self.norm_columns, concatenate con la
    posizione corrente (pi). Il segnale viene generato tramite un processo di Ornstein-Uhlenbeck.
    """
    def __init__(
        self,
        sigma=0.5,
        theta=1.0,
        T=1000,
        random_state=None,
        lambd=0.5,
        psi=0.5,
        cost="trade_0",
        max_pos=10,
        squared_risk=True,
        penalty="none",
        alpha=10,
        beta=10,
        clip=True,
        noise=False,
        noise_std=10,
        noise_seed=None,
        scale_reward=10,
        norm_params_path=None,   # Percorso al file JSON con i parametri Min-Max
        norm_columns=None        # Lista delle colonne (feature) da utilizzare, in ordine
    ):
        self.sigma = sigma
        self.theta = theta
        self.T = T
        self.lambd = lambd
        self.psi = psi
        self.cost = cost
        self.max_pos = max_pos
        self.squared_risk = squared_risk
        self.random_state = random_state
        self.signal = build_ou_process(T, sigma, theta, random_state)
        self.it = 0
        self.pi = 0
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)  # Stato "base" (vecchio formato)
        self.done = False
        self.action_size = 1
        self.penalty = penalty
        self.alpha = alpha
        self.beta = beta
        self.clip = clip
        self.scale_reward = scale_reward
        self.noise = noise
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        if noise:
            if noise_seed is None:
                self.noise_array = np.random.normal(0, noise_std, T)
            else:
                rng = np.random.RandomState(noise_seed)
                self.noise_array = rng.normal(0, noise_std, T)

        # Carica i parametri di normalizzazione, se fornito
        if norm_params_path is not None:
            with open(norm_params_path, 'r') as f:
                self.norm_params = json.load(f)
        else:
            self.norm_params = None

        # Definisci l'ordine esatto delle feature da utilizzare.
        if norm_columns is not None:
            self.norm_columns = norm_columns
        else:
            # Esempio: le 64 feature da utilizzare
            self.norm_columns = [
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
        # La dimensione dello stato è il numero di feature + 1 (per la posizione)
        self.state_size = len(self.norm_columns) + 1

        # Inizializza raw_state come dizionario con chiavi definite da norm_columns.
        # Questo verrà aggiornato ad ogni step con i dati correnti (già normalizzati)
        self.raw_state = {col: 0.0 for col in self.norm_columns}

    def update_raw_state(self, df, current_index):
        """
        Aggiorna self.raw_state leggendo la riga corrente dal DataFrame df.
        Il DataFrame df deve contenere le colonne definite in self.norm_columns.
        """
        row = df.iloc[current_index].to_dict()
        missing = [col for col in self.norm_columns if col not in row]
        if missing:
            raise ValueError(f"Mancano le seguenti colonne nel DataFrame: {missing}")
        self.raw_state = row

    def reset(self, random_state=None, noise_seed=None):
        """
        Resetta l'ambiente per iniziare un nuovo episodio.
        """
        self.signal = build_ou_process(self.T, self.sigma, self.theta, random_state)
        self.it = 0
        self.pi = 0
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)
        self.done = False
        if self.noise:
            if noise_seed is None:
                self.noise_array = np.random.normal(0, self.noise_std, self.T)
            else:
                rng = np.random.RandomState(noise_seed)
                self.noise_array = rng.normal(0, self.noise_std, self.T)
        # Aggiorna raw_state per il nuovo episodio
        # Ad esempio, se hai un DataFrame normalizzato 'df' e un indice corrente '0':
        # self.update_raw_state(df, current_index=0)
        return self.get_state()

    def update_raw_state(self, df, current_index):
        """
        Aggiorna self.raw_state leggendo la riga corrente (current_index) dal DataFrame df.
        Il DataFrame df deve contenere le colonne indicate in self.norm_columns.
        """
        # Estrae la riga come dizionario; questo garantisce che l'ordine non importi
        row = df.iloc[current_index].to_dict()
        # Verifica che tutte le colonne attese siano presenti
        missing = [col for col in self.norm_columns if col not in row]
        if missing:
            raise ValueError(f"Mancano le seguenti colonne nel DataFrame: {missing}")
        self.raw_state = row

    def get_state(self):
        """
        Restituisce lo stato corrente come un vettore ottenuto concatenando le feature
        (estratte da self.raw_state in base a self.norm_columns) e la posizione corrente (pi).

        Questo metodo riordina automaticamente le feature secondo self.norm_columns,
        garantendo che il vettore di stato sia conforme a quello che il modello si aspetta.

        Ritorna:
            Un array NumPy di dimensione (state_size,).
        """
        # Assicura che tutte le chiavi attese siano presenti in raw_state
        missing_cols = [col for col in self.norm_columns if col not in self.raw_state]
        if missing_cols:
            raise ValueError(f"Mancano le seguenti colonne in raw_state: {missing_cols}")

        ordered_features = [self.raw_state[col] for col in self.norm_columns]
        ordered_features.append(self.pi)
        return np.array(ordered_features)

    def step(self, action):
        """
        Applica l'azione all'ambiente, aggiornando la posizione, il segnale e calcolando la ricompensa.

        Parameters:
            action : Float, variazione della posizione (trade).

        Returns:
            Float, la ricompensa ottenuta.
        """
        assert not self.done, "L'episodio è terminato. Resetta l'ambiente prima di procedere."
        pi_next_unclipped = self.pi + action
        if self.clip:
            pi_next = np.clip(self.pi + action, -self.max_pos, self.max_pos)
        else:
            pi_next = self.pi + action

        # Calcola la penalità
        if self.penalty == "none":
            pen = 0
        elif self.penalty == "constant":
            pen = self.alpha * max(
                0,
                (self.max_pos - pi_next) / abs(self.max_pos - pi_next),
                (-self.max_pos - pi_next) / abs(-self.max_pos - pi_next),
            )
        elif self.penalty == "tanh":
            pen = self.beta * (np.tanh(self.alpha * (abs(pi_next_unclipped) - 5 * self.max_pos / 4)) + 1)
        elif self.penalty == "exp":
            pen = self.beta * np.exp(self.alpha * (abs(pi_next) - self.max_pos))

        # Calcola la ricompensa in base al modello di costo
        if self.cost == "trade_0":
            reward = (self.p * pi_next - self.lambd * pi_next ** 2 * self.squared_risk - pen) / self.scale_reward
        elif self.cost == "trade_l1":
            if self.noise:
                reward = ((self.p + self.noise_array[self.it]) * pi_next
                          - self.lambd * pi_next ** 2 * self.squared_risk
                          - self.psi * abs(pi_next - self.pi)
                          - pen) / self.scale_reward
            else:
                reward = (self.p * pi_next
                          - self.lambd * pi_next ** 2 * self.squared_risk
                          - self.psi * abs(pi_next - self.pi)
                          - pen) / self.scale_reward
        elif self.cost == "trade_l2":
            if self.noise:
                reward = ((self.p + self.noise_array[self.it]) * pi_next
                          - self.lambd * pi_next ** 2 * self.squared_risk
                          - self.psi * (pi_next - self.pi) ** 2
                          - pen) / self.scale_reward
            else:
                reward = (self.p * pi_next
                          - self.lambd * pi_next ** 2 * self.squared_risk
                          - self.psi * (pi_next - self.pi) ** 2
                          - pen) / self.scale_reward

        # Aggiorna la posizione e il segnale
        self.pi = pi_next
        self.it += 1
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)  # Stato "base" (per compatibilità)
        self.done = self.it == (len(self.signal) - 2)
        return reward

    # I metodi test() e test_apply() li manteniamo invariati (non li riproduco qui per brevità)


    def test(
        self, agent, model, total_episodes=100, random_states=None, noise_seeds=None
    ):
        """
        Description
        ---------------
        Test a model on a number of simulated episodes and get the average cumulative
        reward.

        Parameters
        ---------------
        agent          : Agent object, the agent that loads the model.
        model          : Actor object, the actor network.
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
            - if None, do not use random state when generating episodes
              (useful to get an idea about the performance of a single model).
            - if List, generate episodes with the values in random_states (useful when
              comparing different models).

        noise_seeds    : None or List of length total_episodes:
                         - if None, do not use a random state when generating the additive
                           noise of the returns
                         - if List, generate noise with seeds in noise_seeds.

        Returns
        ---------------
        2-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        agent.actor_local = model
        if random_states is not None:
            assert total_episodes == len(
                random_states
            ), "random_states should be a list of length total_episodes!"

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                action = agent.act(state, noise=False)
                pi_next = np.clip(self.pi + action, -self.max_pos, self.max_pos)
                episode_positions.append(pi_next)
                reward = self.step(action)
                pnl = reward + (self.lambd * self.pi ** 2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = total_pnl
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #      'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )

    def apply(self, state, thresh=1, lambd=None, psi=None):
        """
        Description
        ---------------
        Apply solution with a certain band and slope outside the band, otherwise apply the
        myopic solution.

        Parameters
        ---------------
        state      : 2-tuple, the current state.
        thresh     : Float>0, price threshold to make a trade.
        lambd      : Float, slope of the solution in the non-banded region.
        psi        : Float, band width of the solution.

        Returns
        ---------------
        Float, the trade to make in state according to this function.
        """

        p, pi = state
        if lambd is None:
            lambd = self.lambd

        if psi is None:
            psi = self.psi

        if not self.squared_risk:
            if abs(p) < thresh:
                return 0
            elif p >= thresh:
                return self.max_pos - pi
            elif p <= -thresh:
                return -self.max_pos - pi

        else:
            if self.cost == "trade_0":
                return p / (2 * lambd) - pi

            elif self.cost == "trade_l2":
                return (p + 2 * psi * pi) / (2 * (lambd + psi)) - pi

            elif self.cost == "trade_l1":
                if p < -psi + 2 * lambd * pi:
                    return (p + psi) / (2 * lambd) - pi
                elif -psi + 2 * lambd * pi <= p <= psi + 2 * lambd * pi:
                    return 0
                elif p > psi + 2 * lambd * pi:
                    return (p - psi) / (2 * lambd) - pi

    def test_apply(
        self,
        total_episodes=10,
        random_states=None,
        thresh=1,
        lambd=None,
        psi=None,
        noise_seeds=None,
        max_point=6.0,
        n_points=1000,
    ):
        """
        Description
        ---------------
        Test a function with certain slope and band width for each reward model (with and
        without trading cost, and depending on the penalty when trading cost is used).
        When psi and lambd are not provided, use the myopic solution.

        Parameters
        ---------------
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
                         - if None, do not use random state when generating episodes
                           (useful to get an idea about the performance of a single
                           model).
                         - if List, generate episodes with the values in random_states
                           (useful when comparing different models).
        lambd          : Float, slope of the solution in the non-banded region.
        psi            : Float, band width of the solution.
        max_point      : Float, the maximum point in the grid [0, max_point]
        n_points       : Int, the number of points in the grid.

        Returns
        ---------------
        5-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
                  - Dict, cumulative sum of the reward at each time step per episode.
                  - Dict, pnl per episode.
                  - Dict, positions per episode.
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        if random_states is not None:
            assert total_episodes == len(
                random_states
            ), "random_states should be a list of length total_episodes!"

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                action = self.apply(state, thresh=thresh, lambd=lambd, psi=psi)
                reward = self.step(action)
                pnl = reward + (self.lambd * self.pi ** 2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                episode_positions.append(state[1])
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = episode_pnls
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #       'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )
