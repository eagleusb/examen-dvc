# Examen DVC et Dagshub

Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen.

```bash
├── examen_dvc
│   ├── data
│   │   ├── processed
│   │   └── raw
│   ├── metrics
│   ├── models
│   │   ├── data
│   │   └── models
│   ├── src
│   └── README.md
```

N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

```mermaid
flowchart TD
        node1["data/raw_data/raw.csv.dvc"]
        node2["evaluate_model"]
        node3["grid_search"]
        node4["normalize_data"]
        node5["split_data"]
        node6["train_model"]
        node1-->node5
        node3-->node6
        node4-->node2
        node4-->node3
        node4-->node6
        node5-->node2
        node5-->node3
        node5-->node4
        node5-->node6
        node6-->node2
```
