"""Les stratégies de jeu : epsilon-greedy, initialisation optimiste, UCB
(Exercices 3, 5 et 6)."""
import math
import random


class GreedyPlayer:
    """Joueur epsilon-greedy (Exercice 3).

    `alpha` : pas d'apprentissage. Par défaut (None) on utilise la moyenne
    empirique, Q += (r - Q) / N, adaptée au cas stationnaire. Un `alpha`
    constant donne plus de poids aux récompenses récentes, utile pour
    l'Exercice 7 (non-stationnaire)."""

    def __init__(self, n=10, eps=0.1, alpha=None):
        self.n = n
        self.eps = eps
        self.alpha = alpha
        self.action_values = [0.0] * n   # Q
        self.eval_count = [0] * n        # N
        self.steps = 0                   # nombre d'appels à reward

    def get_action(self):
        explore = random.random() < self.eps
        if explore:
            return self._random_action()
        return self._greedy_action()

    def _greedy_action(self):
        return self._argmax(self.action_values)

    @staticmethod
    def _argmax(values):
        """Indice de la plus grande valeur, égalités départagées au hasard."""
        best = max(values)
        return random.choice([i for i, v in enumerate(values) if v == best])

    def _random_action(self):
        return random.randrange(self.n)

    def reward(self, action, reward):
        self.steps += 1
        self.eval_count[action] += 1
        step_size = self.alpha if self.alpha is not None else 1 / self.eval_count[action]
        self.action_values[action] += step_size * (reward - self.action_values[action])


class OptimistGreedyPlayer(GreedyPlayer):
    """Greedy avec des estimations initiales optimistes, Q0 = 5
    (Exercice 5). Par défaut eps = 0 : c'est l'optimisme seul qui pousse
    à explorer."""

    def __init__(self, n=10, eps=0.0, initial_value=5.0, alpha=None):
        super().__init__(n, eps, alpha)
        self.action_values = [float(initial_value)] * n


class UCBGreedyPlayer(GreedyPlayer):
    """Sélection Upper-Confidence-Bound (Exercice 6) :

        A_t = argmax_a [ Q(a) + c * sqrt( ln t / N(a) ) ]

    où t est le nombre d'appels à `reward`. Une action jamais essayée
    (N(a) = 0) est considérée comme maximisante."""

    def __init__(self, n=10, c=2.0, eps=0.0, alpha=None):
        super().__init__(n, eps, alpha)
        self.c = c

    def _greedy_action(self):
        untried = [i for i, count in enumerate(self.eval_count) if count == 0]
        if untried:
            return random.choice(untried)
        log_t = math.log(self.steps)
        scores = [q + self.c * math.sqrt(log_t / count)
                  for q, count in zip(self.action_values, self.eval_count)]
        return self._argmax(scores)
