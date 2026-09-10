# TP Morpion — Apprentissage par renforcement

Implémentation du TP 0 - morpion: un agent qui apprend à jouer au
morpion par Temporal-Difference learning, à la Sutton & Barto.

## Structure du projet

| Fichier | Contenu |
|---|---|
| `tic_tac_toe.py` | Exercice 1 — la classe `TicTacToe` (grille, coups, victoire/nul, hash). |
| `opponents.py` | Exercice 2 — `opponent_random` et `opponent_next`. |
| `value_function.py` | Exercice 2.4 — la « value function » (`defaultdict` à 0.5). |
| `agent.py` | Exercice 2.6 / 3.4 — `greedy_move` et `epsilon_greedy_move`. |
| `training.py` | Exercice 2.7 et 3 — `play_game`, `play_game2`, `train`, `evaluate`. |
| `symmetry.py` | Exercice 3.9-10 — réduction de l'espace d'états par symétrie. |
| `main.py` | Interface en ligne de commande (entraînement, évaluation, partie interactive). |

Aucune dépendance externe : uniquement la bibliothèque standard de Python 3.

## Utilisation

```bash
# Exercice 2.8 : vérifier les statistiques aléatoire vs aléatoire
python main.py sanity

# Exercice 3 : entraîner l'agent par TD(0) contre un adversaire, puis
# sauvegarder la value function
python main.py train --opponent random --games 20000
python main.py train --opponent next --games 20000 --symmetry

# Exercice 3.7-3.8 : deux agents RL s'entraînent l'un contre l'autre
python main.py duel --games 20000

# Jouer contre une value function entraînée
python main.py play --model value_function.pkl --human O
```

`python main.py <commande> --help` liste toutes les options (`--alpha`,
`--epsilon`, `--games`, ...).

## Exercice 1 — la classe `TicTacToe`

La grille est une liste à plat de 9 cases (`None`, `"X"` ou `"O"`).
`play`/`undo` sont des opérations strictement inverses l'une de l'autre,
ce qui est ce qui permet à `opponent_next` et `greedy_move` de « regarder »
le résultat d'un coup (jouer, tester, annuler) sans dupliquer la grille.

Point important : `__hash__` et `__eq__` sont tous les deux nécessaires
pour utiliser `TicTacToe` comme clé de dictionnaire (Exercice 2.5). Sans
`__eq__`, deux instances différentes représentant le même état seraient
vues comme deux clés distinctes par un dictionnaire, malgré un hash
identique.

**Piège rencontré et corrigé pendant le développement** : comme le hash
dépend du contenu (mutable) de la grille, il ne faut jamais utiliser comme
clé de dictionnaire un objet `TicTacToe` qui va encore être modifié
ensuite. `greedy_move` teste chaque coup candidat en faisant
`game.play(move)` puis `game.undo(move)` sur l'objet `game` reçu en
argument (donc partagé, muté temporairement). Une première version du code
lisait la valeur avec `value_function[game]` : sur un état jamais vu, le
`defaultdict` insère alors `game` **lui-même** comme clé avec la valeur par
défaut, puis `undo()` change son contenu et donc son hash juste après —
l'entrée devient orpheline dans le dictionnaire (bucket de hash incohérent
avec le contenu actuel de l'objet), ce qui corrompait silencieusement la
value function (des valeurs déjà apprises se retrouvaient écrasées lors
d'une simple évaluation, sans aucun apprentissage censé avoir lieu). La
correction : `greedy_move` utilise `value_function.get(game, ...)`, qui
lit sans jamais insérer. Les seules insertions dans la value function ont
lieu dans `training.py`, sur une `copy.deepcopy(game)` qui n'est plus
jamais modifiée ensuite.

## Exercice 2 — jouer

- `opponent_random` joue un coup légal au hasard.
- `opponent_next` joue un coup gagnant s'il en existe un, sinon un coup au
  hasard (adversaire « myope », un seul coup d'anticipation).
- `greedy_move` évalue chaque coup légal en simulant l'état résultant et
  choisit celui de plus grande valeur (égalité départagée au hasard).
- Vérification demandée en 2.8 (`python main.py sanity`, 10 000 parties
  aléatoire vs aléatoire) :

  ```
  X gagne : 58.5%  (référence : ~59%)
  O gagne : 28.9%  (référence : ~29%)
  Nul     : 12.6%  (référence : ~12%)
  ```

  Conforme à la répartition classique du morpion joué au hasard.

## Exercice 3 — Temporal-Difference

`play_game` fait jouer l'agent RL (toujours X) contre un adversaire fixe.
Après chaque coup de l'agent, on applique la mise à jour TD(0) :

```
V(s_t) <- V(s_t) + alpha * (V(s_{t+1}) - V(s_t))
```

où `s_t` est l'état laissé par le coup précédent de l'agent et `s_{t+1}`
celui qu'il vient d'atteindre (après son propre coup, ou après la réponse
de l'adversaire si celle-ci termine la partie). Les états terminaux
reçoivent directement la vraie issue : `1.0` si X gagne, `0.0` sinon
(défaite ou nul).

Avec `alpha=0.2`, `epsilon` annelé de `0.2` à `0`, sur 5000 parties :

```
avant entraînement (0.5 partout) : ~58% de victoires vs random
après entraînement               : ~98% de victoires vs random et vs next
```

### Réponses aux questions

**3.3 — Que se passe-t-il en 100% greedy dès le départ ?**
Sans exploration, l'agent ne revisite jamais les coups qu'il a écartés
tôt (souvent encore à la valeur par défaut 0.5) une fois qu'un autre coup
a été renforcé au hasard des premières parties. Il n'est pas
*nécessairement* pire dans l'absolu — contre un adversaire aléatoire, il
apprend quand même quelque chose et peut rester correct — mais rien ne
garantit la convergence vers la meilleure stratégie : certaines branches
de l'arbre de jeu peuvent ne jamais être essayées, donc jamais corrigées.
Une **initialisation optimiste** (valeur initiale supérieure à la valeur
réelle attendue, par ex. 1.0 au lieu de 0.5) atténue le problème : tant
qu'un état n'a pas été essayé, il paraît le plus attrayant possible, ce
qui pousse le choix greedy à explorer spontanément tous les coups au
moins une fois. Ce n'est cependant qu'un palliatif *temporaire* : une fois
les valeurs mises à jour, ce biais disparaît, contrairement à
l'exploration epsilon-greedy qui continue à explorer indéfiniment (utile
si l'adversaire change de comportement en cours de route).

**3.5 — Ne pas mettre à jour la value function pour les coups exploratoires ?**
Le paramètre `backup_through_exploration` de `play_game`/`play_game2`
permet de tester les deux variantes :
- En mettant à jour même après un coup exploratoire (`True`, la valeur par
  défaut ici, pour rester fidèle à l'énoncé 3.2 qui demande une mise à
  jour « après chaque coup »), la value function apprend la probabilité de
  victoire **sous la politique effectivement suivie pendant
  l'entraînement**, c'est-à-dire epsilon-greedy (5% de coups aléatoires
  inclus). C'est une estimation légèrement pessimiste par rapport au jeu
  purement greedy, puisqu'elle « absorbe » les conséquences des bourdes
  aléatoires occasionnelles.
- En sautant la mise à jour qui suit un coup exploratoire (`False`, le
  choix fait par Sutton & Barto dans leur exemple original), la value
  function converge vers la probabilité de victoire **sous la politique
  purement greedy** — celle qui sera réellement utilisée une fois
  l'entraînement terminé (`epsilon` ramené à 0).

Le second est généralement préférable si l'objectif final est de déployer
l'agent en jeu purement greedy (ce qui est notre cas : `train` anneale
`epsilon` vers 0) : les valeurs apprises correspondent alors exactement à
la politique qui sera utilisée en pratique, sans bruit d'exploration.

**3.7-3.8 — Deux agents RL identiques l'un contre l'autre**
Contre un adversaire fixe (`opponent_random` ou `opponent_next`), la value
function se spécialise sur les failles précises de cet adversaire (ex. il
apprend à tendre des pièges qu'un joueur aléatoire ne bloque jamais) — un
agent ainsi entraîné n'est pas forcément robuste contre un adversaire plus
malin. En auto-apprentissage (`play_game2`, chaque joueur avec sa propre
value function), la cible de chacun change à chaque partie puisque
l'adversaire apprend aussi (problème classique de cible non stationnaire).

Expérimentalement (`python main.py duel`, 15 000 parties, epsilon annelé
de 0.3 à 0, alpha=0.15, reproductible sur plusieurs graines aléatoires) :
**X finit par gagner quasiment 100% des évaluations en greedy pur, jamais
de match nul.** C'est contre-intuitif si l'on pense que le morpion est
toujours nul avec deux joueurs parfaits, mais c'est un résultat connu et
reproductible du TD(0) en auto-apprentissage naïf : rien ne garantit sa
convergence vers l'équilibre minimax du jeu. X, qui joue toujours en
premier, a un avantage structurel d'initiative ; si sa politique se
stabilise avant que O n'ait exploré suffisamment de parades face à
*toutes* les ouvertures possibles de X, les deux value functions se
co-renforcent autour d'un point d'équilibre où O perd systématiquement
sans jamais rencontrer (donc jamais apprendre à contrer) les lignes de jeu
alternatives dont il aurait besoin. Contrairement à des algorithmes
spécifiquement conçus pour converger vers l'équilibre minimax d'un jeu à
somme nulle, le TD(0) auto-joué « naïf » n'offre aucune garantie de ce
type.

**3.9 — Symétries et Exercice 3.10 — autres améliorations**
Le morpion a 8 symétries (identité, 3 rotations, 4 réflexions) : la grille
vide, par exemple, est en théorie un seul état, pas 8. `symmetry.py`
implémente cette idée : `canonical_board` ramène toute grille à un
représentant canonique unique parmi ses 8 variantes, et
`SymmetricValueFunction` enveloppe la value function pour indexer par ce
représentant. Effet mesuré : sur un même budget d'entraînement, le nombre
d'états distincts appris passe d'environ 950-1300 à **266**, chaque entrée
reçoit donc jusqu'à 8 fois plus de mises à jour pour le même nombre de
parties, donc une convergence plus rapide (`python main.py train
--symmetry`).

Cette hypothèse ne tient que si l'adversaire est lui-même « symétrique »
(indifférent à la position absolue sur la grille). Si l'adversaire
privilégie systématiquement un côté (par ex. le côté droit), alors deux
grilles image l'une de l'autre par symétrie n'ont **plus** la même
probabilité réelle de victoire — les regrouper de force efface une vraie
différence et biaiserait l'apprentissage. La réduction par symétrie est
donc une bonne optimisation contre un adversaire « neutre » (aléatoire, ou
une heuristique elle-même symétrique), mais une source d'erreur contre un
adversaire aux préférences positionnelles.

Autres pistes pour l'Exercice 3.10, évoquées mais non implémentées ici
(hors périmètre du TP, mentionnées pour être complet) :
- **Pas d'apprentissage décroissant** (`alpha` proportionnel à `1/N(s)`,
  le nombre de visites de l'état `s`) plutôt que fixe, pour une
  convergence plus propre une fois un état bien exploré.
- **Entraînement contre un mélange d'adversaires** (random + next +
  auto-jeu) pour éviter la sur-spécialisation contre un seul style
  d'adversaire (cf. réponse 3.7).
- **Parallélisation** des parties d'entraînement (`multiprocessing`) pour
  jouer beaucoup plus de parties dans le même temps.

Deux améliorations sont en revanche implémentées et activables via la CLI :
1. la réduction par symétrie ci-dessus (`--symmetry`) ;
2. la décroissance linéaire d'epsilon au fil de l'entraînement
   (`--epsilon` → `--epsilon-end` dans `train`), qui explore beaucoup au
   début puis joue de plus en plus greedy, accélérant la convergence vers
   une politique forte sans sacrifier la couverture initiale de l'espace
   d'états.
