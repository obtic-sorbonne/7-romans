import argparse, json, pickle
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from renard.ner_utils import ner_entities
from measures import score_ner
from ner import entities_to_BIO

novels = [
    "BelAmi",
    "EugénieGrandet",
    "Germinal",
    "LeRougeEtLeNoir",
    "LesTroisMousquetaires",
    "MadameBovary",
    "NotreDameDeParis",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=str)
    parser.add_argument("-t", "--table", type=str, help="one of: 'classes', 'novels', 'vertices', 'edges'")
    args = parser.parse_args()

    if args.table == "classes":

        with open(f"{args.run}/metrics.json") as f:
            metrics = json.load(f)

        tokens = []
        tags = []
        gold_tags = []

        for novel in tqdm(novels, desc="loading novels..."):

            with open(f"./runs/4/{novel}_state.pickle", "rb") as f:
                state = pickle.load(f)
            tokens += state.tokens
            # NOTE: we convert entities to BIO tags to be able to concatenate
            # novel predictions without hassle. Otherwise, we would have to
            # shift the entities start and end indices.
            tags += entities_to_BIO(state.tokens, state.entities)

            with open(f"./runs/4/{novel}_state_gold.pickle", "rb") as f:
                gold_state = pickle.load(f)
            gold_tags += entities_to_BIO(state.tokens, gold_state.entities)

        entities = ner_entities(tokens, tags)
        gold_entities = ner_entities(tokens, gold_tags)

        ner_classes = {"PER", "ORG", "LOC", "MISC"}
        df_dict = defaultdict(list)
        for cls in tqdm(ner_classes, desc="computing measures..."):
            precision, recall, f1 = score_ner(
                tokens, entities, gold_entities, ignore_classes=ner_classes - {cls}
            )
            df_dict["f1"].append(f1)
            df_dict["precision"].append(precision)
            df_dict["recall"].append(recall)
        df_dict["class"] += list(ner_classes)
        df = pd.DataFrame(df_dict)
        df = df.set_index("class")
        print(
            df.style.format(
                lambda v: "{:.2f}".format(v * 100) if isinstance(v, float) else v
            ).to_latex(hrules=True)
        )

    elif args.table in {"novels", "vertices", "edges"}:

        with open(f"{args.run}/metrics.json") as f:
            metrics = json.load(f)

        if args.table == "novels":
            targets = [
                "F1", "Precision", "Recall"
            ]
        elif args.table == "vertices":
            targets = [
                "VertexF1",
                "VertexPrecision",
                "VertexRecall",
            ]
        elif args.table == "edges":
            targets = [
                "EdgeF1",
                "EdgePrecision",
                "EdgeRecall",
                "WeightedEdgeF1",
                "WeightedEdgePrecision",
                "WeightedEdgeRecall",

            ]
        else:
            raise RuntimeError
        
        df_dict = defaultdict(list)
        for novel in novels:
            df_dict["novel"].append(novel)
            for key in [f"{novel}.{m}" for m in targets]:
                df_dict[key[key.index(".") + 1 :]].append(metrics[key]["values"][0])
        df = pd.DataFrame(df_dict)
        df = df.set_index("novel")
        print(df.style.format(lambda v: "{:.2f}".format(v * 100)).to_latex(hrules=True))


    else:
        raise ValueError(f"unknown table: {args.table}")
