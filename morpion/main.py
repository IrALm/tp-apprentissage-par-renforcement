"""Point d'entrée en ligne de commande : entraînement, évaluation, et
partie interactive.

Exemples
--------
    python main.py sanity
    python main.py train --opponent random --games 20000
    python main.py train --opponent next --symmetry
    python main.py duel --games 20000
    python main.py play --model value_function.pkl --human O
"""
import argparse
import pickle
import sys
from collections import Counter

# Force une sortie UTF-8 : certaines consoles Windows (cp1252) affichent
# sinon les accents des messages français comme des caractères corrompus.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from agent import greedy_move
from opponents import opponent_next, opponent_random
from symmetry import SymmetricValueFunction
from tic_tac_toe import TicTacToe
from training import evaluate, play_game2, play_random_game, train
from value_function import new_value_function

OPPONENTS = {"random": opponent_random, "next": opponent_next}


def cmd_sanity(args):
    """Exercice 2.8 : deux joueurs aléatoires doivent reproduire la
    répartition classique ~59% / ~29% / ~12%."""
    results = Counter(play_random_game() for _ in range(args.games))
    total = args.games
    print(f"Sur {total} parties aléatoire vs aléatoire :")
    print(f"  X gagne : {results['X'] / total:.1%}  (référence : ~59%)")
    print(f"  O gagne : {results['O'] / total:.1%}  (référence : ~29%)")
    print(f"  Nul     : {results[None] / total:.1%}  (référence : ~12%)")


def cmd_train(args):
    opponent = OPPONENTS[args.opponent]
    raw_vf = new_value_function()
    vf = SymmetricValueFunction(raw_vf) if args.symmetry else raw_vf

    print(f"Avant entraînement contre '{args.opponent}' :")
    print(" ", dict(evaluate(vf, opponent, args.eval_games)))

    train(vf, opponent, args.games, alpha=args.alpha, epsilon=args.epsilon,
          epsilon_end=args.epsilon_end)

    print(f"Après {args.games} parties d'entraînement :")
    print(" ", dict(evaluate(vf, opponent, args.eval_games)))
    print(f"Nombre d'états appris : {len(raw_vf)}")

    payload = {"symmetry": args.symmetry, "table": dict(raw_vf)}
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)
    print(f"Value function sauvegardée dans {args.output}")


def cmd_duel(args):
    """Exercice 3.7-3.8 : deux agents RL indépendants, chacun avec sa
    propre value function, s'entraînent l'un contre l'autre."""
    vf_x, vf_o = new_value_function(), new_value_function()
    results = Counter()
    for _ in range(args.games):
        winner = play_game2(vf_x, vf_o, alpha=args.alpha, epsilon=args.epsilon)
        results[winner if winner in ("X", "O") else "draw"] += 1

    print(f"Après {args.games} parties d'auto-apprentissage : {dict(results)}")
    print(f"États appris : X={len(vf_x)}, O={len(vf_o)}")

    with open("value_function_x.pkl", "wb") as f:
        pickle.dump({"symmetry": False, "table": dict(vf_x)}, f)
    with open("value_function_o.pkl", "wb") as f:
        pickle.dump({"symmetry": False, "table": dict(vf_o)}, f)
    print("Sauvegardées dans value_function_x.pkl et value_function_o.pkl")


def _load_value_function(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    vf = new_value_function()
    vf.update(payload["table"])
    return SymmetricValueFunction(vf) if payload["symmetry"] else vf


def cmd_play(args):
    vf = _load_value_function(args.model)

    game = TicTacToe()
    human = args.human
    print(f"Vous jouez {human}. Numérotation des cases :\n 0 | 1 | 2\n 3 | 4 | 5\n 6 | 7 | 8\n")

    while not game.is_over():
        print(game)
        print()
        if game.current_player == human:
            move = int(input(f"Case pour {human} : "))
        else:
            move = greedy_move(vf, game)
            print(f"L'IA ({game.current_player}) joue {move}")
        game.play(move)

    print(game)
    print(f"\nGagnant : {game.winner}" if game.winner else "\nMatch nul")


def build_parser():
    parser = argparse.ArgumentParser(description="TP Morpion - apprentissage par renforcement")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sanity = sub.add_parser("sanity", help="Vérifie les stats aléatoire vs aléatoire (Ex2.8)")
    p_sanity.add_argument("--games", type=int, default=10000)
    p_sanity.set_defaults(func=cmd_sanity)

    p_train = sub.add_parser("train", help="Entraîne l'agent RL par TD(0)")
    p_train.add_argument("--opponent", choices=OPPONENTS, default="random")
    p_train.add_argument("--games", type=int, default=20000)
    p_train.add_argument("--eval-games", type=int, default=2000)
    p_train.add_argument("--alpha", type=float, default=0.1)
    p_train.add_argument("--epsilon", type=float, default=0.1)
    p_train.add_argument("--epsilon-end", type=float, default=0.01)
    p_train.add_argument("--symmetry", action="store_true",
                          help="Réduit l'espace d'états par symétrie (Ex3.9-10)")
    p_train.add_argument("--output", default="value_function.pkl")
    p_train.set_defaults(func=cmd_train)

    p_duel = sub.add_parser("duel", help="Deux agents RL s'entraînent l'un contre l'autre (Ex3.7-3.8)")
    p_duel.add_argument("--games", type=int, default=20000)
    p_duel.add_argument("--alpha", type=float, default=0.1)
    p_duel.add_argument("--epsilon", type=float, default=0.1)
    p_duel.set_defaults(func=cmd_duel)

    p_play = sub.add_parser("play", help="Jouez contre une value function entraînée")
    p_play.add_argument("--model", default="value_function.pkl")
    p_play.add_argument("--human", choices=["X", "O"], default="O")
    p_play.set_defaults(func=cmd_play)

    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
