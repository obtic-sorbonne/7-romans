# Un gold standard pour la reconnaissance d'entités nommées

Ce corpus contient 7 romans entièrement annotés pour la tâche de repérage d'entités nommées et la tâche de résolution d'alias pour les personnages.

| **Roman**               | **Auteur**        | **Année de publication** | **Nombre de tokens** | **Nombre de personnages** |
|-------------------------|-------------------|--------------------------|----------------------|---------------------------|
| Les Trois Mousquetaires | Alexandre Dumas   | 1849                     | 294 989              | 213                       |
| Le Rouge et le Noir     | Stendhal          | 1854                     | 216 445              | 318                       |
| Eugénie Grandet         | Honoré de Balzac  | 1855                     | 80 659               | 107                       |
| Germinal                | Émile Zola        | 1885                     | 220 273              | 102                       |
| Bel-Ami                 | Guy de Maupassant | 1901                     | 138 156              | 150                       |
| Notre-Dame de Paris     | Victor Hugo       | 1904                     | 221 351              | 536                       |
| Madame Bovary           | Gustave Flaubert  | 1910                     | 148 861              | 175                       |


Ce gold standard a été réalisé dans le cadre d'un projet à ObTIC-Sorbonne université, dirigé par Motasem Alrahabi, et annoté par Perrine Morel, Una Faller et Romaric Parnasse.

Le corpus a été ensuite utilisé pour entrainer un nouveau modèle NER, en collaboration avec Arthur Amalvy et Vincent Labatut (université d'Avignon).



# Modèle et reproduction des résultats

Pour reproduire nos résultats concernant notre modèle de REN basé sur CamemBERT, installez d'abord les dépendances Python, soit :

- avec [uv](https://github.com/astral-sh/uv): `uv sync`
- avec `pip`: `pip install -r requirements.txt`

Pour reproduire l'expérience principale, il suffit de lancer le script `xp_main.sh`. Ce script entraîne et évalue notre modèle sur chaque roman du jeu de données par validation croisée. Il évalue également le modèle sur la tâche d'extraction de réseaux de personnages grâce à [Renard](https://github.com/CompNet/Renard). Un GPU avec au moins 8Gb de RAM est conseillé pour l'entraînement. Ce script produit un dossier dans le dossier "runs", contenant tous les résultats (voir notamment le fichier `metrics.json`).

Le script `xp_train.sh` permet d'entraîner le modèle sur les 7 romans (utilisé pour produire le [modèle huggingface](https://huggingface.co/compnet-renard/camembert-base-literary-NER-v2)).



# Citation

Si vous utilisez le corpus dans vos recherches, vous pouvez citer :

```bibtex
@InProceedings{Maurel2025,
  authors = {Maurel, P. and Amalvy, A. and Labatut, V. and Alrahabi, M.},
  title = {Du repérage à l’analyse : un modèle pour la reconnaissance d’entités nommées dans les textes littéraires en français},
  booktitle = {Digital Humanities 2025},
  year = {2025},
}
```
