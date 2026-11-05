"""Simulations et graphiques (Exercices 3 et 4)."""
import os

import matplotlib
matplotlib.use("Agg")  # écriture de PNG, pas besoin de fenêtre
import matplotlib.pyplot as plt
import numpy as np

from player import GreedyPlayer
from snake import ShortSnake

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path)}")
    return path


def run(gamma, n_runs, n_steps, size=4, length=1, eps=0.1, alpha=0.5):
    """Lance n_runs parties indépendantes de n_steps étapes.
    Renvoie la récompense moyenne à chaque étape (moyenne sur les parties)
    et la récompense moyenne par étape de chaque partie."""
    rewards = np.zeros(n_steps)
    per_run = np.zeros(n_runs)
    for i in range(n_runs):
        game = ShortSnake(size, length=length)
        player = GreedyPlayer(game, eps=eps, gamma=gamma, alpha=alpha)
        for t in range(n_steps):
            action = player.get_action()
            reward = game.move(action)
            player.reward(action, reward)
            rewards[t] += reward
            per_run[i] += reward
    return rewards / n_runs, per_run / n_steps


def smooth(values, window):
    """Moyenne glissante sur `window` étapes."""
    if window <= 1:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_curves(curves, title, filename, window=1):
    """Une courbe de récompense moyenne par entrée de `curves` ({label: valeurs})."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, values in curves.items():
        y = smooth(values, window)
        ax.plot(np.arange(len(y)) + window, y, label=label, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Étape")
    ylabel = "Récompense moyenne"
    if window > 1:
        ylabel += f" (moy. glissante {window})"
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return save(fig, filename)


def gamma_search(gammas, n_runs, n_steps, size=4, length=1):
    """Récompense moyenne par étape pour chaque gamma, avec une marge
    d'erreur de 2 écarts-types de la moyenne."""
    means, errors = [], []
    for gamma in gammas:
        _, per_run = run(gamma, n_runs, n_steps, size, length)
        means.append(per_run.mean())
        errors.append(2 * per_run.std() / np.sqrt(n_runs))
        print(f"  gamma = {gamma:.2f} : récompense moyenne = {means[-1]:+.3f} ± {errors[-1]:.3f}")
    return np.array(means), np.array(errors)
