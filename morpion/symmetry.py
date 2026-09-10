"""Exploite la symétrie à 8 éléments (groupe diédral) de la grille de
morpion.

C'est l'« amélioration » évoquée à l'Exercice 3, questions 9 et 10 : la
grille vide, par exemple, est un seul état et non 8 états distincts, et
regrouper les états équivalents par symétrie divise l'espace d'états par
un facteur pouvant aller jusqu'à 8, et permet à l'agent d'apprendre d'un
coup joué dans n'importe quelle orientation.
"""


def _rotate90(board):
    """Fait pivoter la grille 3x3 de 90 degrés dans le sens horaire."""
    return [
        board[6], board[3], board[0],
        board[7], board[4], board[1],
        board[8], board[5], board[2],
    ]


def _mirror(board):
    """Retourne la grille 3x3 horizontalement (effet miroir)."""
    return [
        board[2], board[1], board[0],
        board[5], board[4], board[3],
        board[8], board[7], board[6],
    ]


def all_symmetries(board):
    """Les 8 grilles obtenues à partir de `board` par les symétries du
    carré (4 rotations, chacune avec ou sans effet miroir)."""
    variants = []
    current = list(board)
    for _ in range(4):
        variants.append(current)
        variants.append(_mirror(current))
        current = _rotate90(current)
    return variants


def canonical_board(board):
    """Un représentant canonique, commun à toutes les grilles équivalentes
    à `board` par rotation/symétrie."""
    return min(
        (tuple(variant) for variant in all_symmetries(board)),
        key=lambda t: tuple(cell or "." for cell in t),
    )


class SymmetricValueFunction:
    """Enveloppe une value function classique pour que les grilles
    équivalentes par symétrie partagent une seule et même entrée.

    S'utilise exactement comme un dictionnaire indexé par des instances de
    TicTacToe (comme l'exigent greedy_move / play_game), par exemple
    `vf[game]`, `vf[game] = 1.0`, `vf.get(game)`.
    """

    def __init__(self, value_function):
        self.table = value_function

    def _key(self, game):
        return canonical_board(game.board), game.current_player

    def __getitem__(self, game):
        return self.table[self._key(game)]

    def __setitem__(self, game, value):
        self.table[self._key(game)] = value

    def __contains__(self, game):
        return self._key(game) in self.table

    def __len__(self):
        return len(self.table)

    def get(self, game, default=None):
        # Comme self.table est un defaultdict, .get() ne déclenche pas son
        # usine par défaut : c'est une lecture pure, sans insertion.
        return self.table.get(self._key(game), default)
