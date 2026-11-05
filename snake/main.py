"""Point d'entrée : une commande par exercice, les figures sont écrites
dans figures/.

Exemples
--------
    python main.py ex1
    python main.py ex3
    python main.py all
"""
import argparse
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import matplotlib.pyplot as plt
import numpy as np

from experiments import gamma_search, plot_curves, run, save
from player import GreedyPlayer
from snake import ShortSnake

GAMMAS = [0, 0.1, 0.5, 0.9, 1]


def cmd_ex1(args):
    """Exercice 1 : démonstration du jeu sur une grille 5 x 5."""
    game = ShortSnake(5)
    game.fruit_pos = (1, 0)  # fruit placé à la main pour la démonstration
    print(game, "\n")
    for direction in [None, None, None, "haut", "bas"]:
        reward = game.move(direction)
        print(f"move({direction!r}) -> récompense {reward:+d}")
        print(game, "\n")


ARROWS = {"haut": "↑", "bas": "↓", "gauche": "←", "droite": "→"}


def draw_state(ax, x, y, state, name, n=3, cell=0.3, fruit_known=True):
    """Dessine un état comme une petite grille n x n centrée en (x, y)."""
    body, direction, fruit = state
    x0, y0 = x - n * cell / 2, y + n * cell / 2
    for r in range(n):
        for c in range(n):
            color = "white"
            text = ""
            if (r, c) == body[0]:
                color, text = "#9fd39f", ARROWS[direction]
            elif fruit_known and (r, c) == fruit:
                color, text = "#f4a0a0", "F"
            ax.add_patch(plt.Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell,
                                       facecolor=color, edgecolor="gray"))
            if text:
                ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, text,
                        ha="center", va="center", fontsize=11)
    ax.text(x, y0 + 0.15, name, ha="center", va="bottom", fontsize=12, weight="bold")


def cmd_ex2(args):
    """Exercice 2 : nombre d'états et schéma du MDP sur une grille 3 x 3."""
    for n, m in [(10, 10), (4, 4), (3, 3)]:
        print(f"  grille {n} x {m} : {n * m} x 4 x {n * m - 1} = {n * m * 4 * (n * m - 1)} états")

    # Les transitions du schéma sont calculées avec le vrai jeu
    # (restore_state puis move), pour être sûrs qu'elles sont justes.
    fruit = (0, 2)
    states = {
        "A": (((1, 0),), "droite", fruit),
        "B": (((1, 1),), "droite", fruit),
        "C": (((1, 2),), "droite", fruit),
        "D": (((0, 1),), "haut", fruit),
    }
    game = ShortSnake(3)
    edges = []
    for start, action in [("A", "droite"), ("B", "droite"), ("C", "droite"),
                          ("B", "haut"), ("D", "droite")]:
        game.restore_state(states[start])
        reward = game.move(action)
        new_state = game.save_state()
        end = next((k for k, s in states.items() if s == new_state), "E")
        if end == "E":
            states["E"] = new_state
        edges.append((start, end, action, reward))
        print(f"  {start} --{action}--> {end}  récompense {reward:+d}")

    pos = {"A": (0, 0), "B": (4, 0), "C": (8, 0), "D": (4, 4), "E": (8, 4)}
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, (x, y) in pos.items():
        draw_state(ax, x, y, states[name], name, fruit_known=(name != "E"))
    ax.text(8, 3.25, "nouveau fruit au hasard\n(8 cases possibles, p = 1/8)",
            ha="center", va="top", fontsize=9)
    arrow = dict(arrowstyle="-|>", color="black", lw=1.2, shrinkA=0, shrinkB=0)
    for start, end, action, reward in edges:
        (x1, y1), (x2, y2) = pos[start], pos[end]
        if start == "C" and end == "A":   # retour par le bord : on dessine un arc
            ax.annotate("", xy=(x2, y2 - 0.6), xytext=(x1, y1 - 0.6),
                        arrowprops=dict(arrow, connectionstyle="arc3,rad=-0.35"))
            ax.text(4, -2.3, f"{action}, r = {reward}  (passage par le bord)",
                    ha="center", fontsize=10, color="#b00000")
        elif y1 == y2:
            ax.annotate("", xy=(x2 - 0.6, y2), xytext=(x1 + 0.6, y1), arrowprops=arrow)
            ax.text((x1 + x2) / 2, y1 + 0.15, f"{action}, r = {reward:+d}".replace("+0", "0"),
                    ha="center", fontsize=10)
        else:
            ax.annotate("", xy=(x2, y2 - 0.6), xytext=(x1, y1 + 1.0), arrowprops=arrow)
            ax.text(x1 + 0.15, (y1 + y2) / 2, f"{action}, r = {reward:+d}".replace("+0", "0"),
                    ha="left", fontsize=10)
    ax.set_xlim(-1.5, 9.5)
    ax.set_ylim(-3, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Ex 2.7 : quelques états du MDP (grille 3 x 3, fruit F en (0, 2))")
    save(fig, "ex2_7_mdp.png")


def play(game, player, n_steps):
    """Fait jouer le joueur n_steps étapes et renvoie la liste des récompenses."""
    rewards = []
    for t in range(n_steps):
        action = player.get_action()
        reward = game.move(action)
        player.reward(action, reward)
        rewards.append(reward)
    return rewards


def plot_one_game(rewards, title, filename):
    """Récompense de chaque étape (en haut) et score cumulé (en bas)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(rewards, ".", markersize=3)
    ax1.set_ylabel("Récompense")
    ax1.set_title(title)
    ax2.plot(np.cumsum(rewards))
    ax2.set_ylabel("Score cumulé")
    ax2.set_xlabel("Étape")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, filename)


def cmd_ex3_10x10(args):
    """Exercice 3 sur la grille 10 x 10 de la question 2.2."""
    print(" une partie de 1000 étapes (gamma = 0.5)")
    game = ShortSnake(10)
    rewards = play(game, GreedyPlayer(game, gamma=0.5), 1000)
    print(f"  score final = {game.score}")
    plot_one_game(rewards, "Grille 10 x 10 : une partie, gamma = 0.5",
                  "ex3_10x10_une_partie.png")

    for n_steps, n_runs, window in [(1000, 500, 20), (10000, 100, 200)]:
        print(f" différents gamma, {n_steps} étapes, {n_runs} parties")
        curves = {}
        for g in GAMMAS:
            curve, per_run = run(g, n_runs, n_steps, size=10)
            curves[f"gamma = {g}"] = curve
            print(f"  gamma = {g} : récompense moyenne = {per_run.mean():+.3f}")
        plot_curves(curves, f"Grille 10 x 10 : facteur d'actualisation, {n_steps} étapes",
                    f"ex3_10x10_gamma_{n_steps}.png", window=window)

    print(" entraînement long : 200000 étapes, 5 parties (gamma = 0.5)")
    curve, _ = run(0.5, 5, 200000, size=10)
    for start in range(0, 200000, 40000):
        print(f"  étapes {start:6d} à {start + 40000:6d} : récompense moyenne = "
              f"{curve[start:start + 40000].mean():+.3f}")
    plot_curves({"grille 10 x 10": curve}, "Grille 10 x 10 : 200000 étapes, gamma = 0.5",
                "ex3_10x10_200000.png", window=5000)


def cmd_ex3(args):
    """Exercice 3 : Q-Learning sur un serpent d'une case."""
    size = args.size

    print(" 3.10 : une seule partie de 1000 étapes (gamma = 0.5)")
    game = ShortSnake(size)
    rewards = play(game, GreedyPlayer(game, gamma=0.5), 1000)
    print(f"  score final = {game.score}")
    plot_one_game(rewards, f"Ex 3.10 : une partie, grille {size} x {size}, gamma = 0.5",
                  "ex3_10_une_partie.png")

    print(" 3.11 : combien de parties pour lisser la courbe ?")
    curves = {}
    for n_runs in [10, 100, 500, 2000]:
        curve, _ = run(0.5, n_runs, 1000, size)
        curves[f"{n_runs} parties"] = curve
        print(f"  {n_runs:5d} parties : écart-type de la courbe sur les 100 dernières "
              f"étapes = {curve[-100:].std():.3f}")
    plot_curves(curves, "Ex 3.11 : moyenne sur plusieurs parties (gamma = 0.5)",
                "ex3_11_nb_parties.png")

    print(" 3.12 : différents gamma, 1000 étapes, 500 parties")
    curves = {f"gamma = {g}": run(g, 500, 1000, size)[0] for g in GAMMAS}
    plot_curves(curves, "Ex 3.12 : facteur d'actualisation, 1000 étapes",
                "ex3_12_gamma_1000.png", window=20)

    print(" 3.13 : recherche du meilleur gamma, 1000 étapes, 500 parties")
    gammas = np.round(np.arange(0, 1.01, 0.1), 2)
    means_1000, err_1000 = gamma_search(gammas, 500, 1000, size)

    print(" 3.14 : 10000 étapes, 100 parties")
    curves = {f"gamma = {g}": run(g, 100, 10000, size)[0] for g in GAMMAS}
    plot_curves(curves, "Ex 3.14 : facteur d'actualisation, 10000 étapes",
                "ex3_14_gamma_10000.png", window=200)
    means_10000, err_10000 = gamma_search(gammas, 100, 10000, size)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(gammas, means_1000, yerr=err_1000, marker="o", capsize=3, label="1000 étapes")
    ax.errorbar(gammas, means_10000, yerr=err_10000, marker="o", capsize=3, label="10000 étapes")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Récompense moyenne par étape")
    ax.set_title("Ex 3.13-3.14 : recherche du meilleur gamma")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "ex3_13_recherche_gamma.png")

    print(" 3.14 : plus grande valeur Q au cours d'une partie (gamma = 0.9 et 1)")
    for g in [0.9, 1]:
        game = ShortSnake(size)
        player = GreedyPlayer(game, gamma=g)
        for step in [10000, 20000, 30000, 40000, 50000]:
            play(game, player, 10000)
            print(f"  gamma = {g} : après {step} étapes, Q max = "
                  f"{max(player.action_values.values()):.1f}")


def cmd_ex4(args):
    """Exercice 4 (bonus) : serpents plus longs."""
    size = args.size
    gamma = args.gamma

    print(" 4.5 : serpent de 2 cases, différents gamma")
    curves = {f"gamma = {g}": run(g, 500, 1000, size, length=2)[0] for g in GAMMAS}
    plot_curves(curves, "Ex 4.5 : serpent de 2 cases, 1000 étapes",
                "ex4_5_gamma_1000.png", window=20)
    curves = {f"gamma = {g}": run(g, 100, 10000, size, length=2)[0] for g in GAMMAS}
    plot_curves(curves, "Ex 4.5 : serpent de 2 cases, 10000 étapes",
                "ex4_5_gamma_10000.png", window=200)

    print(f" 4.6 : 1 case contre 2 cases (gamma = {gamma})")
    curves = {}
    for length in [1, 2]:
        curve, per_run = run(gamma, 200, 10000, size, length=length)
        curves[f"longueur {length}"] = curve
        print(f"  longueur {length} : récompense moyenne = {per_run.mean():+.3f}")
    plot_curves(curves, f"Ex 4.6 : 1 case contre 2 cases (gamma = {gamma})",
                "ex4_6_longueur_1_2.png", window=200)

    print(f" 4.2 / 4.7 : nombre d'états sur une grille {size} x {size}")
    for length in range(1, 7):
        shapes = 4 if length == 1 else 4 * 3 ** (length - 2)
        n_states = size * size * shapes * (size * size - length)
        print(f"  longueur {length} : au plus {n_states} états")

    print(f" 4.7 : longueurs 1 à 6, 50000 étapes (gamma = {gamma})")
    curves = {}
    for length in range(1, 7):
        curve, per_run = run(gamma, 20, 50000, size, length=length)
        curves[f"longueur {length}"] = curve
        print(f"  longueur {length} : récompense moyenne sur les 5000 dernières "
              f"étapes = {curve[-5000:].mean():+.3f}")
    plot_curves(curves, f"Ex 4.7 : serpents plus longs (gamma = {gamma})",
                "ex4_7_longueurs.png", window=1000)


COMMANDS = {"ex1": cmd_ex1, "ex2": cmd_ex2, "ex3_10x10": cmd_ex3_10x10,
            "ex3": cmd_ex3, "ex4": cmd_ex4}


def main():
    parser = argparse.ArgumentParser(description="TP MDP : Snake")
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--size", type=int, default=4, help="taille de la grille (défaut 4)")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma pour ex4 (défaut 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="graine aléatoire (défaut 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        print(f"== {name} ==")
        COMMANDS[name](args)


if __name__ == "__main__":
    main()
