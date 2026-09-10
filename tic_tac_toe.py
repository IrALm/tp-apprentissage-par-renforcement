"""Moteur du jeu de morpion (Exercice 1).

Cette classe gère uniquement la grille et le tour de jeu. L'interaction
avec un humain ou une IA est de la responsabilité de l'appelant
(main.py, training.py, ...).
"""
from __future__ import annotations

# Les 8 alignements gagnants, sous forme de triplets d'indices sur une
# grille à plat 0..8 disposée en lecture de gauche à droite, de haut en bas :
#   0 | 1 | 2
#   3 | 4 | 5
#   6 | 7 | 8
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # lignes
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colonnes
    (0, 4, 8), (2, 4, 6),             # diagonales
)

OTHER_PLAYER = {"X": "O", "O": "X"}


class CellOccupiedError(Exception):
    """Levée quand on essaie de jouer sur une case déjà occupée."""


class TicTacToe:
    """Une grille de morpion. X commence toujours."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Réinitialise la grille à vide, X commence."""
        self.board = [None] * 9
        self.current_player = "X"

    def play(self, position):
        """Joue `current_player` sur `position` (0-8), puis passe la main."""
        if self.board[position] is not None:
            raise CellOccupiedError(f"La case {position} est déjà occupée")
        self.board[position] = self.current_player
        self.current_player = OTHER_PLAYER[self.current_player]

    def undo(self, position):
        """Défait le coup joué en `position` (doit être le dernier coup joué)."""
        self.board[position] = None
        self.current_player = OTHER_PLAYER[self.current_player]

    @property
    def allowed_moves(self):
        """Liste des indices de cases encore libres."""
        return [i for i, cell in enumerate(self.board) if cell is None]

    @property
    def winner(self):
        """Renvoie 'X', 'O' ou None."""
        for a, b, c in WIN_LINES:
            if self.board[a] is not None and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def has_winner(self):
        return self.winner is not None

    def is_draw(self):
        return self.winner is None and not self.allowed_moves

    def is_over(self):
        return self.has_winner() or self.is_draw()

    def __str__(self):
        symbols = [cell or " " for cell in self.board]
        rows = [" | ".join(symbols[r * 3:(r + 1) * 3]) for r in range(3)]
        return f"\n{'-' * 9}\n".join(rows)

    def __repr__(self):
        return f"TicTacToe(board={self.board!r}, current_player={self.current_player!r})"

    def __eq__(self, other):
        if not isinstance(other, TicTacToe):
            return NotImplemented
        return self.board == other.board and self.current_player == other.current_player

    def __hash__(self):
        # Un état de jeu est entièrement déterminé par le contenu de la
        # grille et le joueur dont c'est le tour. On construit une chaîne
        # de caractères représentant les deux, puis on la hash. Il faut
        # aussi définir __eq__ ci-dessus : sans lui, un dictionnaire
        # traiterait deux instances différentes ayant la même grille comme
        # deux clés distinctes.
        #
        # Attention : comme ce hash dépend du contenu mutable de la grille,
        # il ne faut JAMAIS utiliser comme clé de dictionnaire une instance
        # encore susceptible d'être modifiée (play/undo) par la suite. Voir
        # agent.greedy_move, qui prend soin de ne lire que des valeurs
        # (via .get) sans jamais insérer une clé dans cet état intermédiaire.
        signature = "".join(cell or "." for cell in self.board) + self.current_player
        return hash(signature)
