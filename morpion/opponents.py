"""Adversaires simples, sans apprentissage, utilisés pour entraîner et
évaluer l'agent RL (Exercice 2)."""
import random


def opponent_random(game):
    """Joue un coup légal uniformément au hasard."""
    return random.choice(game.allowed_moves)


def opponent_next(game):
    """Joue un coup gagnant s'il y en a un ce tour-ci, sinon joue au
    hasard. Cet adversaire ne regarde qu'un seul coup à l'avance."""
    for move in game.allowed_moves:
        game.play(move)
        wins = game.has_winner()
        game.undo(move)
        if wins:
            return move
    return random.choice(game.allowed_moves)
