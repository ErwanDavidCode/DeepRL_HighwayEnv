import time
import optuna
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Charger la configuration
from configuration3 import config_dict

# Fonction pour créer l'environnement
def make_env():
    env = gym.make("intersection-v0")
    env.unwrapped.configure(config_dict)
    return Monitor(env)

# Fonction d'optimisation
def optimize_ppo(trial):
    # Hyperparamètres à optimiser
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_uniform("gamma", 0.8, 0.999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-5, 1e-2)
    clip_range = trial.suggest_uniform("clip_range", 0.1, 0.4)

    # Créer l'environnement
    env = DummyVecEnv([make_env])

    # Créer le modèle PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0,
    )

    # Entraîner le modèle
    model.learn(total_timesteps=10000)

    # Évaluer le modèle
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

    # Fermer l'environnement
    env.close()

    return mean_reward

# Callback pour afficher le temps restant
class TimeCallback:
    def __init__(self, n_trials):
        self.start_time = None
        self.n_trials = n_trials

    def __call__(self, study, trial):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed_time = time.time() - self.start_time
        completed_trials = trial.number + 1
        avg_time_per_trial = elapsed_time / completed_trials
        remaining_trials = self.n_trials - completed_trials
        estimated_time_remaining = avg_time_per_trial * remaining_trials
        print(f"Trial {completed_trials}/{self.n_trials} - Temps écoulé : {elapsed_time:.2f}s - Temps restant estimé : {estimated_time_remaining:.2f}s")

# Activer les logs d'Optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

# Nombre de trials
n_trials = 30

# Étudier les hyperparamètres avec Optuna
time_callback = TimeCallback(n_trials)
study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=n_trials, callbacks=[time_callback])

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres :", study.best_params)