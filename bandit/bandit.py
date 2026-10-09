"""Les bandits manchots : un bras (Bandit), dix bras (Ban10), et leurs
variantes non-stationnaires (Exercices 1, 2 et 7)."""
import math
import random


class Bandit:
    """Un bras de machine à sous : récompense ~ N(avg, 1), avec
    avg ~ N(0, 1) tiré une fois pour toutes à la construction (Exercice 1)."""

    def __init__(self):
        self.avg = random.gauss(0, 1)

    def play(self):
        return random.gauss(self.avg, 1)


class Ban10:
    """Bandit à 10 bras (Exercice 2)."""

    bandit_class = Bandit

    def __init__(self, n=10):
        self.bandits = [self.bandit_class() for _ in range(n)]
        self.best_action = self._compute_best_action()

    def _compute_best_action(self):
        avgs = [b.avg for b in self.bandits]
        return avgs.index(max(avgs))

    def play(self, action):
        return self.bandits[action].play()


class MovingBandit(Bandit):
    """Bandit dont la moyenne oscille sinusoïdalement : elle avance d'un pas
    à chaque appel de `play` (Exercice 7.1). Période, min, max et phase
    sont tirés au hasard."""

    def __init__(self):
        low, high = sorted((random.gauss(0, 1), random.gauss(0, 1)))
        self.center = (low + high) / 2
        self.amplitude = (high - low) / 2
        self.period = random.uniform(100, 500)
        self.phase = random.uniform(0, 2 * math.pi)
        self.t = 0
        self._update_avg()

    def _update_avg(self):
        self.avg = self.center + self.amplitude * math.sin(
            2 * math.pi * self.t / self.period + self.phase)

    def step(self):
        """Fait avancer le temps d'un coup sans jouer."""
        self.t += 1
        self._update_avg()

    def play(self):
        reward = super().play()
        self.step()
        return reward


class MovingBan10(Ban10):
    """Ban10 composé de MovingBandit (Exercice 7.2).

    Choix de conception : à chaque coup sur la machine, *tous* les bras
    avancent d'un pas (le bras joué via `play`, les autres via `step`).
    Sinon un bras jamais joué resterait figé, ce qui supprimerait
    justement la difficulté du non-stationnaire : un bras délaissé peut
    devenir le meilleur sans que l'agent le voie."""

    bandit_class = MovingBandit

    def play(self, action):
        for i, bandit in enumerate(self.bandits):
            if i != action:
                bandit.step()
        reward = self.bandits[action].play()
        self.best_action = self._compute_best_action()
        return reward
