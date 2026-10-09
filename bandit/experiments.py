"""Simulations et graphiques (Exercices 1, 3 et 4 à 7)."""
import copy
import os

import matplotlib
matplotlib.use("Agg")  # écriture de PNG, pas besoin de fenêtre
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Ordre fixe des couleurs de série (jamais recyclé) + encres neutres.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_PRIMARY,
    "lines.linewidth": 1.5,
    "font.size": 10,
})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path)}")
    return path


def run_experiment(player_factories, ban_class, n_runs=2000, n_steps=1000):
    """Fait jouer chaque stratégie sur `n_runs` problèmes, étape par étape
    (les n_runs instances jouent leur 1er coup, puis leur 2e, etc.,
    Exercice 4.2).

    `player_factories` : {label: fonction sans argument -> Player}.
    Pour chaque problème on crée UN bandit, puis chaque stratégie joue sur
    une copie de ce même bandit (mêmes moyennes de départ) : les
    stratégies sont comparées sur exactement les mêmes problèmes
    (Exercice 5.3). La copie est nécessaire car un MovingBan10 évolue à
    chaque coup : partager l'objet ferait avancer son temps deux fois.

    Renvoie {label: (récompense_moyenne[t], %_optimal[t])}."""
    bans = [ban_class() for _ in range(n_runs)]
    results = {}
    for label, make_player in player_factories.items():
        games = [(copy.deepcopy(ban), make_player()) for ban in bans]
        rewards = np.zeros(n_steps)
        optimal = np.zeros(n_steps)
        for t in range(n_steps):
            reward_sum = 0.0
            optimal_count = 0
            for ban, player in games:
                best = ban.best_action  # lu avant le coup (MovingBan10 bouge)
                action = player.get_action()
                reward = ban.play(action)
                player.reward(action, reward)
                reward_sum += reward
                optimal_count += action == best
            rewards[t] = reward_sum / n_runs
            optimal[t] = 100 * optimal_count / n_runs
        results[label] = (rewards, optimal)
        print(f"  {label:<28} récompense moy. (100 derniers coups) = "
              f"{rewards[-100:].mean():.3f}   % optimal = {optimal[-100:].mean():.1f}%")
    return results


def plot_results(results, title, filename):
    """Deux panneaux empilés : récompense moyenne et % de choix optimal.
    Deux graphiques séparés plutôt qu'un double axe Y."""
    fig, (ax_r, ax_o) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    end_labels = []
    for color, (label, (rewards, optimal)) in zip(SERIES_COLORS, results.items()):
        ax_r.plot(np.arange(1, len(rewards) + 1), rewards, color=color, label=label)
        ax_o.plot(np.arange(1, len(optimal) + 1), optimal, color=color, label=label)
        end_labels.append([optimal[-50:].mean(), label])
    # étiquettes directes en bout de courbe, écartées d'au moins 4.5 points
    # de % pour ne pas se chevaucher quand les courbes finissent proches
    end_labels.sort()
    for below, above in zip(end_labels, end_labels[1:]):
        above[0] = max(above[0], below[0] + 4.5)
    for y, label in end_labels:
        ax_o.annotate(label, (len(optimal), y), xytext=(6, 0),
                      textcoords="offset points", va="center",
                      fontsize=8, color=TEXT_PRIMARY, annotation_clip=False)
    ax_r.set_ylabel("Récompense moyenne")
    ax_o.set_ylabel("% de choix optimal")
    ax_o.set_xlabel("Coup")
    ax_o.set_ylim(0, 100)
    ax_r.set_title(title, loc="left", fontsize=12)
    ax_r.legend(loc="lower right")
    fig.tight_layout()
    return save(fig, filename)
