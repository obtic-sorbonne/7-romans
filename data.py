from collections import defaultdict
from typing import Literal, Tuple, List
import pandas as pd
from datasets import Dataset
from more_itertools import flatten
from renard.ner_utils import hgdataset_from_conll2002, ner_entities
from renard.pipeline.core import Mention
from renard.pipeline.character_unification import Character

NovelTitle = Literal[
    "BelAmi",
    "EugénieGrandet",
    "Germinal",
    "LeRougeEtLeNoir",
    "LesTroisMousquetaires",
    "NotreDameDeParis",
]

NER_ID2LABEL = {
    0: "B-LOC",
    1: "B-MISC",
    2: "B-ORG",
    3: "B-PER",
    4: "I-LOC",
    5: "I-MISC",
    6: "I-ORG",
    7: "I-PER",
    8: "O",
}


def load_novel(novel_name: NovelTitle) -> Tuple[Dataset, List[Character]]:
    """Load NER and character unification gold data for a novel

    :param novel_name: name of the novel to load

    :return: (NER Huggingface Dataset, list of characters)
    """
    ner_dataset = hgdataset_from_conll2002(f"./ner/{novel_name}.conll", separator=" ")
    per_entities = [
        e
        for e in ner_entities(
            list(flatten(ner_dataset["tokens"])),
            [NER_ID2LABEL[l] for l in flatten(ner_dataset["labels"])],
        )
        if e.tag == "PER"
    ]

    alias_df = pd.read_csv(f"./alias-resolution/{novel_name}.csv")
    char_dict = defaultdict(set)
    for _, row in alias_df.iterrows():
        char_dict[row["Entity"]].add(row["Form"])
    characters = []
    for names in char_dict.values():
        mentions = [
            Mention(e.tokens, e.start_idx, e.end_idx)
            for e in per_entities
            if " ".join(e.tokens) in names
        ]
        characters.append(Character(frozenset(names), mentions))

    return (ner_dataset, characters)
