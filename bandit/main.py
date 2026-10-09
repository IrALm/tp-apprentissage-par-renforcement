"""Point d'entrée : une commande par exercice, les figures sont écrites
dans figures/.

Exemples
--------
    python main.py ex1
    python main.py ex3
    python main.py ex4 --runs 2000 --steps 1000
    python main.py all
"""
import argparse
import random
import sys

# Sortie UTF-8 pour les accents sur les consoles Windows (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import matplotlib.pyplot as plt
import numpy as np

from bandit import Ban10, Bandit, MovingBan10
from experiments import SERIES_COLORS, plot_results, run_experiment, save
from players import GreedyPlayer, OptimistGreedyPlayer, UCBGreedyPlayer


def cmd_ex1(args):
    """Exercice 1.3-1.4 : histogrammes de 1000 tirages, 1 puis 3 bandits."""
    bins = np.linspace(-5, 5, 41)
    fig, ax = plt.subplots(figsize=(8, 4))
    bandit = Bandit()
    ax.hist([bandit.play() for _ in range(1000)], bins=bins,
            color=SERIES_COLORS[0], edgecolor="#fcfcfb", linewidth=1)
    ax.axvline(bandit.avg, color="#0b0b0b", linewidth=1, linestyle="--")
    ax.set_title(f"Ex 1.3 — 1000 tirages d'un Bandit (avg = {bandit.avg:.2f})",
                 loc="left")
    ax.set_xlabel("Récompense")
    ax.set_ylabel("Effectif")
    save(fig, "ex1_3_un_bandit.png")
    print(f"  Bandit : avg = {bandit.avg:.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, color in enumerate(SERIES_COLORS[:3]):
        b = Bandit()
        samples = [b.play() for _ in range(1000)]
        ax.hist(samples, bins=bins, color=color, alpha=0.55,
                label=f"Bandit {i} (avg = {b.avg:.2f})")
        ax.axvline(b.avg, color=color, linewidth=1.5, linestyle="--")
        print(f"  Bandit {i} : avg = {b.avg:.3f}, moyenne empirique = {np.mean(samples):.3f}")
    ax.set_title("Ex 1.4 — 1000 tirages de 3 Bandit", loc="left")
    ax.set_xlabel("Récompense")
    ax.set_ylabel("Effectif")
    ax.legend()
    save(fig, "ex1_4_trois_bandits.png")


def cmd_ex2(args):
    """Exercice 2 : affiche les moyennes d'un Ban10 et son meilleur bras."""
    ban = Ban10()
    for i, b in enumerate(ban.bandits):
        mark = "  <- meilleur" if i == ban.best_action else ""
        print(f"  bras {i} : avg = {b.avg:+.3f}{mark}")
    print(f"  best_action = {ban.best_action}, play(best_action) = "
          f"{ban.play(ban.best_action):+.3f}")


def cmd_ex3(args):
    """Exercice 3.9-3.11 + 4.1 : un joueur epsilon-greedy, 1000 coups."""
    ban = Ban10()
    player = GreedyPlayer(10, args.eps)
    rewards = []
    n_optimal = 0
    for t in range(args.steps):
        action = player.get_action()
        reward = ban.play(action)
        player.reward(action, reward)
        rewards.append(reward)
        optimal = action == ban.best_action
        n_optimal += optimal
        if args.verbose or t < 20 or t >= args.steps - 5:
            print(f"  coup {t:4d} : bras {action}  récompense {reward:+.3f}  "
                  f"{'optimal' if optimal else '-'}")
        elif t == 20:
            print("  ...")
    print(f"  Meilleur bras : {ban.best_action}. Choix optimal sur "
          f"{n_optimal}/{args.steps} coups.")
    print(f"  Q estimés : {[round(q, 2) for q in player.action_values]}")
    print(f"  Vrais avg : {[round(b.avg, 2) for b in ban.bandits]}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(rewards) + 1), rewards, color=SERIES_COLORS[0], linewidth=0.8)
    ax.set_title(f"Ex 4.1 — Récompense d'un seul GreedyPlayer (ε = {args.eps})",
                 loc="left")
    ax.set_xlabel("Coup")
    ax.set_ylabel("Récompense")
    save(fig, "ex4_1_un_joueur.png")


def cmd_ex4(args):
    """Exercice 4.2-4.5 : moyenne sur 2000 problèmes, eps = 0, 0.01, 0.1."""
    results = run_experiment(
        {f"ε = {eps}": (lambda eps=eps: GreedyPlayer(10, eps)) for eps in (0, 0.01, 0.1)},
        Ban10, args.runs, args.steps)
    plot_results(results, f"Ex 4 — ε-greedy, moyenne sur {args.runs} problèmes",
                 "ex4_epsilon_greedy.png")


def cmd_ex5(args):
    """Exercice 5 : optimiste (Q0 = 5, eps = 0) vs eps-greedy 0.1."""
    results = run_experiment({
        "ε-greedy ε = 0.1, Q0 = 0": lambda: GreedyPlayer(10, 0.1),
        "Optimiste ε = 0, Q0 = 5": lambda: OptimistGreedyPlayer(10, 0.0, 5.0),
        # Avec la moyenne empirique, Q0 est entièrement oublié au 1er tirage
        # (pas de 1/1) ; avec un pas constant il s'estompe progressivement.
        "Optimiste ε = 0, Q0 = 5, α = 0.1":
            lambda: OptimistGreedyPlayer(10, 0.0, 5.0, alpha=0.1),
    }, Ban10, args.runs, args.steps)
    plot_results(results, "Ex 5 — Initialisation optimiste vs ε-greedy",
                 "ex5_optimiste.png")


def cmd_ex6(args):
    """Exercice 6 : UCB (c = 2) vs eps-greedy 0.1."""
    results = run_experiment({
        "ε-greedy ε = 0.1": lambda: GreedyPlayer(10, 0.1),
        f"UCB c = {args.c}": lambda: UCBGreedyPlayer(10, args.c),
    }, Ban10, args.runs, args.steps)
    plot_results(results, "Ex 6 — UCB vs ε-greedy", "ex6_ucb.png")


def cmd_ex7(args):
    """Exercice 7 (bonus) : bandits non-stationnaires."""
    print(" 7.3 — différentes valeurs de ε (moyenne empirique)")
    results = run_experiment(
        {f"ε = {eps}": (lambda eps=eps: GreedyPlayer(10, eps)) for eps in (0, 0.01, 0.1, 0.2)},
        MovingBan10, args.runs, args.steps)
    plot_results(results, "Ex 7.3 — MovingBan10, ε-greedy", "ex7_3_epsilon.png")

    print(" 7.4 — optimiste et UCB (moyenne empirique)")
    results = run_experiment({
        "ε-greedy ε = 0.1": lambda: GreedyPlayer(10, 0.1),
        "Optimiste Q0 = 5": lambda: OptimistGreedyPlayer(10, 0.0, 5.0),
        f"UCB c = {args.c}": lambda: UCBGreedyPlayer(10, args.c),
    }, MovingBan10, args.runs, args.steps)
    plot_results(results, "Ex 7.4 — MovingBan10, optimiste et UCB",
                 "ex7_4_optimiste_ucb.png")

    print(" 7 (extension) — pas constant α = 0.1 au lieu de la moyenne empirique")
    results = run_experiment({
        "ε = 0.1, moyenne empirique": lambda: GreedyPlayer(10, 0.1),
        "ε = 0.1, α = 0.1": lambda: GreedyPlayer(10, 0.1, alpha=0.1),
        "Optimiste Q0 = 5, α = 0.1": lambda: OptimistGreedyPlayer(10, 0.0, 5.0, alpha=0.1),
        f"UCB c = {args.c}, α = 0.1": lambda: UCBGreedyPlayer(10, args.c, alpha=0.1),
    }, MovingBan10, args.runs, args.steps)
    plot_results(results, "Ex 7 — MovingBan10, pas constant α = 0.1",
                 "ex7_alpha_constant.png")


COMMANDS = {"ex1": cmd_ex1, "ex2": cmd_ex2, "ex3": cmd_ex3, "ex4": cmd_ex4,
            "ex5": cmd_ex5, "ex6": cmd_ex6, "ex7": cmd_ex7}


def main():
    parser = argparse.ArgumentParser(description="TP bandit à N bras")
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--runs", type=int, default=2000, help="nombre de problèmes (défaut 2000)")
    parser.add_argument("--steps", type=int, default=1000, help="coups par problème (défaut 1000)")
    parser.add_argument("--eps", type=float, default=0.1, help="ε pour ex3 (défaut 0.1)")
    parser.add_argument("--c", type=float, default=2.0, help="c pour UCB (défaut 2)")
    parser.add_argument("--seed", type=int, default=42, help="graine aléatoire (défaut 42)")
    parser.add_argument("--verbose", action="store_true", help="ex3 : afficher les 1000 coups")
    args = parser.parse_args()

    random.seed(args.seed)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        print(f"== {name} ==")
        COMMANDS[name](args)


if __name__ == "__main__":
    main()
