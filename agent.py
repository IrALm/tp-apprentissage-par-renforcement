"""Sélection du coup à jouer par l'agent RL, à partir d'une value function
(Exercice 2 & 3)."""
import random

from value_function import UNKNOWN_STATE_VALUE


def greedy_move(value_function, game):
    """Renvoie le coup légal menant à l'état de plus grande valeur.
    En cas d'égalité, le choix se fait uniformément au hasard parmi les
    coups ex-æquo (Exercice 2.6)."""
    best_value = None
    best_moves = []
    for move in game.allowed_moves:
        game.play(move)
        # Important : on lit la valeur avec .get() et pas avec value_function[game].
        # `game` est ici temporairement modifié puis annulé juste après (undo) ;
        # comme TicTacToe.__hash__ dépend du contenu de la grille, l'insérer
        # comme clé pendant cet état transitoire puis le muter à nouveau
        # corromprait le dictionnaire (sa position de hash changerait sous
        # ses pieds). .get() se contente de lire, sans jamais insérer.
        value = value_function.get(game, UNKNOWN_STATE_VALUE)
        game.undo(move)

        if best_value is None or value > best_value:
            best_value = value
            best_moves = [move]
        elif value == best_value:
            best_moves.append(move)

    return random.choice(best_moves)


def epsilon_greedy_move(value_function, game, epsilon):
    """Avec une probabilité `epsilon`, joue un coup uniformément aléatoire
    (« exploratoire ») plutôt que le coup greedy (Exercice 3.4).
    Renvoie (coup, était_exploratoire)."""
    if epsilon > 0 and random.random() < epsilon:
        return random.choice(game.allowed_moves), True
    return greedy_move(value_function, game), False
