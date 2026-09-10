"""Boucles de jeu (Exercice 2 & 3).

`play_game` implémente l'apprentissage par Temporal-Difference décrit dans
l'exemple du morpion de Sutton & Barto : l'agent RL joue toujours X, choisit
ses coups par epsilon-greedy à partir de `value_function`, et après chacun
de ses coups rapproche la valeur de l'état qu'il vient de quitter de la
valeur de l'état qu'il vient d'atteindre. Les états terminaux reçoivent
directement leur véritable issue (1 = victoire, 0 = défaite/nul).
"""
import copy
from collections import Counter

from agent import epsilon_greedy_move
from opponents import opponent_random
from tic_tac_toe import TicTacToe


def play_random_game(opponent_x=opponent_random, opponent_o=opponent_random):
    """Joue une partie complète entre deux stratégies sans apprentissage,
    sans aucune value function (Exercice 2.3 et 2.8)."""
    game = TicTacToe()
    while not game.is_over():
        move = opponent_x(game) if game.current_player == "X" else opponent_o(game)
        game.play(move)
    return game.winner  # 'X', 'O' ou None (nul)


def play_game(value_function, opponent, *, alpha=0.1, epsilon=0.0, learn=True,
              backup_through_exploration=True):
    """Joue une partie complète : l'agent RL (X) contre `opponent` (O).

    alpha : pas d'apprentissage de la mise à jour TD.
    epsilon : probabilité de jouer un coup exploratoire (aléatoire) plutôt
        que le coup greedy.
    learn : si False, la value function n'est que lue, jamais mise à jour
        (utilisé pour l'évaluation).
    backup_through_exploration : si False, ne met pas à jour la valeur de
        l'état qui précédait un coup exploratoire (Exercice 3.5).

    Renvoie 'X', 'O' ou None (match nul).
    """
    game = TicTacToe()
    prev_state = None       # état juste après le coup précédent de l'agent RL
    prev_exploratory = False
    mover = None

    while not game.is_over():
        mover = game.current_player

        if mover == "X":
            move, exploratory = epsilon_greedy_move(value_function, game, epsilon)
            game.play(move)
            # On travaille sur une copie indépendante : contrairement à
            # `game` (mutable et réutilisé pendant toute la partie), cette
            # copie ne sera plus jamais modifiée, ce qui en fait une clé de
            # dictionnaire sûre (voir la remarque dans agent.greedy_move).
            state = copy.deepcopy(game)

            if learn:
                if prev_state is not None and (backup_through_exploration or not prev_exploratory):
                    value_function[prev_state] += alpha * (value_function[state] - value_function[prev_state])

                if game.has_winner():
                    value_function[state] = 1.0
                elif game.is_draw():
                    value_function[state] = 0.0

            prev_state, prev_exploratory = state, exploratory
        else:
            move = opponent(game)
            game.play(move)
            if learn and prev_state is not None and (game.has_winner() or game.is_draw()):
                if backup_through_exploration or not prev_exploratory:
                    # O a gagné ou complété un match nul : le dernier état
                    # laissé par l'agent RL n'a pas mené à une victoire,
                    # on rapproche sa valeur de 0.
                    value_function[prev_state] += alpha * (0.0 - value_function[prev_state])

    return mover if game.has_winner() else None


def play_game2(value_function_x, value_function_o, *, alpha=0.1, epsilon=0.05,
               learn=True, backup_through_exploration=True):
    """Même principe que play_game, mais les DEUX joueurs sont des agents
    RL, chacun avec sa propre value function (Exercice 3.7-3.8)."""
    game = TicTacToe()
    value_functions = {"X": value_function_x, "O": value_function_o}
    prev_state = {"X": None, "O": None}
    prev_exploratory = {"X": False, "O": False}
    mover = None

    while not game.is_over():
        mover = game.current_player
        other = "O" if mover == "X" else "X"
        vf = value_functions[mover]

        move, exploratory = epsilon_greedy_move(vf, game, epsilon)
        game.play(move)
        state = copy.deepcopy(game)

        if learn:
            if prev_state[mover] is not None and (backup_through_exploration or not prev_exploratory[mover]):
                vf[prev_state[mover]] += alpha * (vf[state] - vf[prev_state[mover]])

            if game.has_winner() or game.is_draw():
                vf[state] = 1.0 if game.has_winner() else 0.0
                if prev_state[other] is not None:
                    other_vf = value_functions[other]
                    if backup_through_exploration or not prev_exploratory[other]:
                        # Le joueur courant vient de terminer la partie :
                        # du point de vue de l'autre joueur, ce n'est
                        # jamais une victoire.
                        other_vf[prev_state[other]] += alpha * (0.0 - other_vf[prev_state[other]])

        prev_state[mover], prev_exploratory[mover] = state, exploratory

    return mover if game.has_winner() else None


def train(value_function, opponent, n_games, *, alpha=0.1, epsilon=0.1, epsilon_end=None,
          **kwargs):
    """Entraîne `value_function` pendant `n_games` parties contre `opponent`.

    Si `epsilon_end` est fourni, epsilon décroît linéairement de `epsilon`
    vers `epsilon_end` au fil de l'entraînement (Exercice 3.10 : on
    explore beaucoup au début puis de plus en plus « greedy » ensuite, ce
    qui accélère la convergence).
    """
    results = Counter()
    for i in range(n_games):
        if epsilon_end is not None and n_games > 1:
            eps = epsilon + (epsilon_end - epsilon) * i / (n_games - 1)
        else:
            eps = epsilon
        winner = play_game(value_function, opponent, alpha=alpha, epsilon=eps, **kwargs)
        results[winner if winner in ("X", "O") else "draw"] += 1
    return results


def evaluate(value_function, opponent, n_games, **kwargs):
    """Joue n_games parties entièrement en greedy, sans apprentissage, pour
    mesurer la performance actuelle."""
    kwargs.setdefault("epsilon", 0.0)
    kwargs["learn"] = False
    results = Counter()
    for _ in range(n_games):
        winner = play_game(value_function, opponent, **kwargs)
        results[winner if winner in ("X", "O") else "draw"] += 1
    return results
