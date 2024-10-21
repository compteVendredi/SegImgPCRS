# Segmentation d'images aériennes

Projet de segmentation d'images aériennes en vue d'obtenir des PCRS.

## Pré-requis

Avoir une carte graphique supportant CUDA (>=12.1) et avec au moins 12go de mémoires.

## Installation

Suivre installation.sh sinon l'exécuter pour créer un environnement nommé lightning dans miniconda
qui contient tout le nécessaire pour exécuter le projet. Il est possible que Tensorboard ne
fonctionne pas, dans ce cas suivre le commentaire dans installation.sh pour le faire fonctionner.
Si ça ne fonctionne pas (sans doute parce que les différentes bibliothèques ont évoluées
entre-temps) regardez dans req.txt pour récupérer les bonnes versions.

## (optionnel) Préparation des données

Dans le dossier utils se trouve les différents fichiers qui ont permis de construire le dossier
data/, comme ils dépendent de la structure des données en entrée ils sont très peu lisibles
(puisque on en a plus besoin après). On peut s'intéresser au fichier final_create_dataset.py

## Les données

Les jeux de données et les données sont dans le dossier data/, pour exécuter le projet il est nécessaire d'avoir les différents jeux de données et les fichiers de configuration camréras présents dans les données.

Ce qui se fait en téléchargeant les données dans les liens suivants et en les placants dans le dossier data/

## (optionnel) Entraînement et résultats

Pour entraîner et récupérer les résultats des modèles il suffit d'exécuter main.py en étant bien dans l'environnement miniconda que vous avez crée plus haut. Voir le fichier main.py pour plus
d'informations.

## Visualisation des résultats

Télécharger le dossier logs dans le lien suivant \[...\].
Faites dans la racine du projet (en étant bien dans le bon environnement) :
> tensorboard --logdir logs/

