"""Le jeu : variante de Snake où le serpent ne grandit pas et ne meurt pas
(Exercice 1, et Exercice 4 pour les serpents plus longs)."""
import random

DIRECTIONS = {"haut": (-1, 0), "bas": (1, 0), "gauche": (0, -1), "droite": (0, 1)}
OPPOSITE = {"haut": "bas", "bas": "haut", "gauche": "droite", "droite": "gauche"}


def legal_actions(direction):
    """Les directions possibles : toutes sauf le demi-tour."""
    return [d for d in DIRECTIONS if d != OPPOSITE[direction]]


class ShortSnake:
    def __init__(self, n=10, m=None, length=1):
        self.n = n                          # nombre de lignes
        self.m = m if m is not None else n  # nombre de colonnes
        self.grid = [["." for _ in range(self.m)] for _ in range(self.n)]
        # le serpent part du milieu, vers la droite ; body[0] est la tête
        self.snake_pos = (self.n // 2, self.m // 2)
        self.snake_dir = "droite"
        self.body = [(self.snake_pos[0], (self.snake_pos[1] - i) % self.m)
                     for i in range(length)]
        self.fruit_pos = self._new_fruit()
        self.score = 0

    def _new_fruit(self):
        """Case aléatoire qui n'est pas occupée par le serpent."""
        free = [(r, c) for r in range(self.n) for c in range(self.m)
                if (r, c) not in self.body]
        return random.choice(free)

    def move(self, direction=None):
        """Avance d'une case (en tournant si `direction` le demande) et
        renvoie la récompense de ce coup."""
        if direction is not None and direction != OPPOSITE[self.snake_dir]:
            self.snake_dir = direction
        dr, dc = DIRECTIONS[self.snake_dir]
        row, col = self.snake_pos[0] + dr, self.snake_pos[1] + dc
        reward = 0
        if not (0 <= row < self.n and 0 <= col < self.m):
            # bord atteint : on réapparaît de l'autre côté
            row, col = row % self.n, col % self.m
            reward -= 10
        self.snake_pos = (row, col)
        self.body = [self.snake_pos] + self.body[:-1]
        if self.snake_pos in self.body[1:]:
            # le serpent se mord la queue (possible à partir de 5 cases)
            reward -= 10
        if self.snake_pos == self.fruit_pos:
            reward += 10
            self.fruit_pos = self._new_fruit()
        self.score += reward
        return reward

    def save_state(self):
        return (tuple(self.body), self.snake_dir, self.fruit_pos)

    def restore_state(self, state):
        body, self.snake_dir, self.fruit_pos = state
        self.body = list(body)
        self.snake_pos = self.body[0]

    def __str__(self):
        for r in range(self.n):
            for c in range(self.m):
                self.grid[r][c] = "."
        self.grid[self.fruit_pos[0]][self.fruit_pos[1]] = "F"
        for r, c in self.body[1:]:
            self.grid[r][c] = "o"
        self.grid[self.snake_pos[0]][self.snake_pos[1]] = "@"
        lines = [" ".join(row) for row in self.grid]
        lines.append(f"direction : {self.snake_dir}   score : {self.score}")
        return "\n".join(lines)
