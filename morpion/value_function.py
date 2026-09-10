"""La « value function » est un dictionnaire qui associe à un état du jeu
l'estimation courante de la probabilité de victoire de l'agent RL depuis
cet état. Les états inconnus valent 0.5 par défaut (Exercice 2.4)."""
from collections import defaultdict

# Valeur par défaut d'un état jamais rencontré : ni victoire, ni défaite.
UNKNOWN_STATE_VALUE = 0.5


def _unknown_state_value():
    # Une vraie fonction plutôt qu'un lambda, pour que le defaultdict reste
    # sérialisable avec pickle (Exercice 3.6) : pickle ne sait pas
    # sérialiser les lambdas.
    return UNKNOWN_STATE_VALUE


def new_value_function():
    return defaultdict(_unknown_state_value)
