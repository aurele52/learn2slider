# Learn2Slither

Projet de reinforcement learning : un Snake contrôlé par
un agent Q-learning.

Deux implémentations sont disponibles :
- Q-table : apprentissage tabulaire.
- DQN : Deep Q-Network avec PyTorch.

## Installation

Python 3.13 et un environnement graphique sont recommandés.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pour un système sans GPU, PyTorch peut être installé
depuis son index CPU :

```bash
python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cpu
```

## Lancement

```bash
python snake.py
```

Ouvre le lobby et le panneau de configuration.

Le menu permet de choisir :
- L'agent Q-table ou DQN.
- Un modèle enregistré.
- Les dimensions du plateau.
- La vitesse.
- Le nombre de sessions.
- L'entraînement ou le mode sans apprentissage.
- Le mode pas à pas.

## Commandes de démonstration

Q-table sans apprentissage :

```bash
python snake.py -agent qtable \
  -load models/qtable-best.pkl \
  -dontlearn -visual on
```

DQN sans apprentissage :

```bash
python snake.py -agent dqn \
  -load models/dqn-best.pth \
  -dontlearn -visual on
```

DQN sur un plateau de 20x20 :

```bash
python snake.py -agent dqn \
  -load models/dqn-best.pth \
  -dontlearn -width 20 -height 20
```

Mode pas à pas :

```bash
python snake.py -agent dqn \
  -load models/dqn-best.pth \
  -dontlearn -step-by-step
```

Commandes du jeu :
- N : avancer d'un mouvement en mode pas à pas.
- Espace : basculer entre pas à pas et automatique.
- Échap : fermer la partie.

## Entraînement

Entraîner une Q-table :

```bash
python snake.py -agent qtable \
  -sessions 100 \
  -visual off \
  -save models/new-qtable.pkl
```

Entraîner un DQN :

```bash
python snake.py -agent dqn \
  -sessions 1000 \
  -visual off \
  -save models/new-dqn.pth
```

Reprendre l'entraînement DQN :

```bash
python snake.py -agent dqn \
  -load models/dqn-best.pth \
  -sessions 1000 \
  -visual off \
  -save models/dqn-resumed.pth
```

## Évaluation

```bash
python evaluation.py -agent qtable \
  -load models/qtable-best.pkl -sessions 100
```

```bash
python evaluation.py -agent dqn \
  -load models/dqn-best.pth -sessions 100
```

## Architecture

- snake.py : arguments CLI et lancement.
- game.py : déroulement des sessions.
- trainer.py : sauvegardes et checkpoints.
- evaluation.py : mesure des performances.
- policy.py : décision Q-table sans apprentissage.
- agent.py : agent Q-table.
- dqn_agent.py : agent Deep Q-Network.
- environement.py : règles du plateau.
- interpreter.py : vision et récompenses.
- tile.py : types de cases.
- utils.py : directions.
- lobby.py : menu et configuration.
- hud.py : affichage des statistiques.

L'agent utilise uniquement les observations de la vision
du serpent, dans les quatre directions.

## Modèles

Le répertoire models/ contient notamment trois Q-tables
entraînées pendant 1, 10 et 100 sessions, ainsi que les
modèles de démonstration Q-table et DQN.

## Bonus

- Longueur supérieure à 35 segments.
- Dimensions du plateau configurables.
- Lobby, configuration et panneau de statistiques.

Les performances dépendent du modèle et des conditions
d'évaluation.
