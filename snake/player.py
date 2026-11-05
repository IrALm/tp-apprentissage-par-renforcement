"""Le joueur : Q-Learning avec une stratégie epsilon-greedy, repris du
GreedyPlayer du TP bandit (Exercice 3)."""
import random

from snake import legal_actions


class GreedyPlayer:
    def __init__(self, game, eps=0.1, gamma=0.5, alpha=0.5):
        self.game = game          # instance de ShortSnake
        self.eps = eps
        self.gamma = gamma        # facteur d'actualisation
        self.alpha = alpha        # taux d'apprentissage
        self.action_values = {}   # Q[(état, action)], 0 si absent
        self.state = None         # état dans lequel l'action a été choisie

    def q(self, state, action):
        return self.action_values.get((state, action), 0.0)

    def get_action(self):
        self.state = self.game.save_state()
        explore = random.random() < self.eps
        if explore:
            return self._random_action()
        return self._greedy_action()

    def _greedy_action(self):
        actions = legal_actions(self.state[1])
        values = [self.q(self.state, a) for a in actions]
        best = max(values)
        return random.choice([a for a, v in zip(actions, values) if v == best])

    def _random_action(self):
        return random.choice(legal_actions(self.state[1]))

    def reward(self, action, reward):
        """Mise à jour du Q-Learning :
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))"""
        new_state = self.game.save_state()
        best_next = max(self.q(new_state, a) for a in legal_actions(new_state[1]))
        old = self.q(self.state, action)
        self.action_values[(self.state, action)] = old + self.alpha * (
            reward + self.gamma * best_next - old)
