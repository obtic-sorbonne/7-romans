from collections import defaultdict
from typing import Literal, Tuple, List, Dict
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
    "MadameBovary",
    "NotreDameDeParis",
]

NERLabel = Literal[
    "B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"
]

NER_ID2LABEL: Dict[int, NERLabel] = {
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


def instances_nb(ner_dataset: Dataset, ner_label: NERLabel) -> int:
    nb = 0
    for labels in ner_dataset["labels"]:
        for label in labels:
            # NOTE: we use 'get' to account for -100 padding labels
            if NER_ID2LABEL.get(label) == ner_label:
                nb += 1
    return nb


def load_novel(novel_name: NovelTitle) -> Tuple[Dataset, List[Character]]:
    """Load NER and character unification gold data for a novel

    :param novel_name: name of the novel to load

    :return: (NER Huggingface Dataset, list of characters)
    """
    # HACK: we split paragraphs so that they fit into 512
    # tokens. Experimentally, we noted that the original text was 0.77
    # smaller than the wordpiece-tokenized text. hence, we cut
    # paragraphs in chunks of 370 tokens (a bit less than 0.77 * 512)
    ner_dataset = hgdataset_from_conll2002(
        f"./ner/{novel_name}.conll", separator=" ", max_sent_len=370
    )
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
        if row["Entity"] == "?":
            continue
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
