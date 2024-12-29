# -*- eval: (code-cells-mode); -*-
# %% Load raw JSON
from __future__ import annotations
from typing import List, Tuple, Literal, Set, Optional, Dict
import json

with open("./7 romans/7 romans.json") as f:
    raw_data = json.load(f)


# %% Load dataset ft. pydantic
from pydantic import BaseModel


class AnnotationResultValue(BaseModel):
    start: int
    end: int
    labels: List[str]
    text: Optional[str] = None


class AnnotationResult(BaseModel):
    id: str
    type: str
    value: AnnotationResultValue


class AnnotationCompletedBy(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str


class Annotation(BaseModel):
    id: int
    completed_by: AnnotationCompletedBy
    result: List[AnnotationResult]


class DatasetExample(BaseModel):
    id: int
    data: Dict[Literal["text"], Optional[str]]
    annotations: List[Annotation]
    # post addition
    novel: Optional[str] = None


class Dataset(BaseModel):
    examples: List[DatasetExample]


dataset = Dataset(examples=raw_data)

# examples are ordered as follows:
# - example 1 from novel 1
# - ...
# - example n from novel 1
# - example 1 from novel 2
# - ...
# - ...
# - example n from novel n
#
# based on the ID of the first example of each novel, we assign a
# "novel" attribute to examples for later below.
startid_to_novel = {
    105605374: "BelAmi",
    105609058: "EugénieGrandet",
    105610222: "Germinal",
    105613403: "LeRougeEtLeNoir",
    105616881: "LesTroisMousquetaires",
    105625469: "MadameBovary",
    105628464: "NotreDameDeParis",
}
current_novel = "BelAmi"
assert startid_to_novel[dataset.examples[0].id] == "BelAmi"
for example in dataset.examples:
    current_novel = startid_to_novel.get(example.id, current_novel)
    example.novel = current_novel


# %% Parse annotations into CoNLL format
import re
from tqdm import tqdm
from sacremoses import MosesTokenizer, MosesDetokenizer
from dataclasses import dataclass


class SpanTokenizer:
    """Adapted from https://gist.github.com/psorianom/32d54b743f3e3bf9c7ee6417ef8b042e"""

    def __init__(self):
        self.tokenizer = MosesTokenizer(lang="fr")
        self.detokenizer = MosesDetokenizer(lang="fr")

    def __call__(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        tokens = self.tokenizer.tokenize(text, escape=False)

        token_spans = []
        tail = text
        offset = 0
        for token in tokens:
            detok = self.detokenizer.detokenize(
                tokens=[token], return_str=True, unescape=False
            )
            m = re.search(re.escape(detok), tail)
            start, end = m.span()
            token_spans.append(
                (
                    detok,
                    start + offset,
                    end + offset,
                )
            )
            offset += end
            tail = tail[end:]

        return token_spans


@dataclass
class ConllAnnotation:
    tokens: List[str]
    tags: List[str]
    annotator_id: Optional[int]
    annotator_email: Optional[str]
    novel: str


def ex_to_conll(
    ex: DatasetExample, span_tokenizer: SpanTokenizer
) -> List[ConllAnnotation]:
    # [(token, cstart, cend), ...]
    spans = span_tokenizer(ex.data["text"])
    tokens = [span[0] for span in spans]

    conll_annotations = []

    for annotation in ex.annotations:

        tags = ["O"] * len(tokens)

        for result in annotation.result:
            assert len(result.value.labels) == 1

            # this is pretty lenient, but what about the case where
            # tokenization does not respect an annotator choice?
            # ex: Prie-Dieu
            #     [O  ][PER]
            # but Prie-Dieu is a single token per Moses!
            start, end = (result.value.start, result.value.end)
            ent_spans = [
                i
                for i, s in enumerate(spans)
                if (start < s[2] and end >= s[2]) or (start <= s[1] and end > s[1])
            ]

            if len(ent_spans) == 0:
                print(f"warning: ignoring malformated result {result.id}")
                continue

            tags[ent_spans[0]] = f"B-{result.value.labels[0]}"
            for i in ent_spans[1:]:
                tags[i] = f"I-{result.value.labels[0]}"

        assert len(tokens) == len(tags)
        assert not ex.novel is None
        conll_annotations.append(
            ConllAnnotation(
                tokens,
                tags,
                annotation.completed_by.id,
                annotation.completed_by.email,
                ex.novel,
            )
        )

    return conll_annotations


tokenizer = SpanTokenizer()
conll_annotations = [
    ex_to_conll(ex, tokenizer)
    for ex in tqdm(dataset.examples, desc="converting to CoNLL")
]


# %% Annotator agreement
import itertools as it
import pandas as pd
from statistics import mean
from seqeval.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

# get all annotators
annotators = set()
for annotations in conll_annotations:
    for annotation in annotations:
        annotators.add(annotation.annotator_id)


def agreement(
    annotators: Set[int],
    conll_annotations: List[List[ConllAnnotation]],
    ner_class: Literal["all", "PERS", "ORG", "LOC", "MISC"],
) -> Tuple[float, float]:
    """
    :return: f1, cohen kappa on annotated tokens
    """
    if ner_class == "PERS":
        # normalization: the model predict PER, not
        # PERS, but the name is PERS in the article
        ner_class = "PER"  # type: ignore

    def filter_tags(tags: List[str]) -> List[str]:
        if ner_class == "all":
            return tags
        return [t if t[2:] == ner_class else "O" for t in tags]

    fscore_lst = []
    kappa_lst = []

    for annotator1, annotator2 in it.combinations(annotators, 2):

        ann1_annotations = []
        ann2_annotations = []

        # Only keep annotations common to both annotators
        for annotations in conll_annotations:
            try:
                annotation1 = next(
                    a for a in annotations if a.annotator_id == annotator1
                )
                annotation2 = next(
                    a for a in annotations if a.annotator_id == annotator2
                )
            except StopIteration:
                continue
            ann1_annotations.append(annotation1)
            ann2_annotations.append(annotation2)

        # F1-score
        f1 = f1_score(
            [filter_tags(a.tags) for a in ann1_annotations],
            [filter_tags(a.tags) for a in ann2_annotations],
        )
        fscore_lst.append(f1)

        # Kappa on annotated tokens only
        tags1 = []
        tags2 = []
        for annotation1, annotation2 in zip(ann1_annotations, ann2_annotations):
            annot1_tags = filter_tags(annotation1.tags)
            annot2_tags = filter_tags(annotation2.tags)
            atleast1_annot = [
                t1 != "O" or t2 != "O" for t1, t2 in zip(annot1_tags, annot2_tags)
            ]
            tags1 += [t for t, atleast1 in zip(annot1_tags, atleast1_annot) if atleast1]
            tags2 += [t for t, atleast1 in zip(annot2_tags, atleast1_annot) if atleast1]
        assert len(tags1) == len(tags2)
        kappa = cohen_kappa_score(tags1, tags2)
        kappa_lst.append(kappa)

    return mean(fscore_lst), mean(kappa_lst)


# compute agreement metrics
print("Computing agreement metrics...", end="")
f1, kappa = agreement(annotators, conll_annotations, "all")
f1_per, kappa_per = agreement(annotators, conll_annotations, "PERS")
f1_loc, kappa_loc = agreement(annotators, conll_annotations, "LOC")
f1_org, kappa_org = agreement(annotators, conll_annotations, "ORG")
f1_misc, kappa_misc = agreement(annotators, conll_annotations, "MISC")
print("done!")
print(f"Agreement F1-score: {f1}")
print(f"Cohen's Kappa on annotated tokens only: {kappa}")
df = pd.DataFrame(
    {
        "class": ["PERS", "LOC", "ORG", "MISC"],
        "F1-score": [f1_per, f1_loc, f1_org, f1_misc],
        "Kappa": [kappa_per, kappa_loc, kappa_org, kappa_misc],
    }
)
print(df)

# %% Agreement matrices
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from more_itertools import flatten


def find_annotator_name(
    annotator: int, conll_annotations: List[List[ConllAnnotation]]
) -> str:
    for annotation in flatten(conll_annotations):
        if (
            annotation.annotator_id == annotator
            and not annotation.annotator_email is None
        ):
            cut_index = re.search(r"[\.@]", annotation.annotator_email).start()
            return annotation.annotator_email[:cut_index]
    raise ValueError


def amatshow(
    ax,
    annotators: Set[int],
    conll_annotations: List[List[ConllAnnotation]],
    metric: Literal["F1", "Kappa"],
    ner_class: Literal["all", "PERS", "ORG", "LOC", "MISC"],
):
    fontsize = 18
    print(f"computing agreement for metric {metric} on class {ner_class}...", end="")
    sannotators = sorted(annotators)
    M = np.ones((len(annotators), len(annotators)))
    for annotator1, annotator2 in it.combinations(sannotators, 2):
        x = sannotators.index(annotator1)
        y = sannotators.index(annotator2)
        f1, kappa = agreement({annotator1, annotator2}, conll_annotations, ner_class)
        M[x][y] = f1 if metric == "F1" else kappa
        M[y][x] = f1 if metric == "F1" else kappa

    ax.matshow(M, vmin=0, vmax=1, cmap=matplotlib.cm.gray)
    for (i, j), value in np.ndenumerate(M):
        ax.text(
            j,
            i,
            "{:.2f}".format(value * 100),
            ha="center",
            va="center",
            fontsize=fontsize,
        )
    ax.set_xticks(list(range(len(sannotators))))
    ax.set_yticks(list(range(len(sannotators))))
    annotator_names = [f"annotateur {i+1}" for i in range(len(sannotators))]
    ax.set_xticklabels(annotator_names, rotation=45, fontsize=fontsize)
    ax.set_yticklabels(annotator_names, fontsize=fontsize)
    if ner_class == "all":
        ner_class = "Général"
    ax.set_title(f"{ner_class}", fontsize=fontsize)
    print("done!")


fig = plt.figure()
gs = gridspec.GridSpec(2, 6)
amatshow(plt.subplot(gs[0, 0:2]), annotators, conll_annotations, "F1", "PERS")
amatshow(plt.subplot(gs[0, 2:4]), annotators, conll_annotations, "F1", "LOC")
amatshow(plt.subplot(gs[0, 4:6]), annotators, conll_annotations, "F1", "ORG")
amatshow(plt.subplot(gs[1, 1:3]), annotators, conll_annotations, "F1", "MISC")
amatshow(plt.subplot(gs[1, 3:5]), annotators, conll_annotations, "F1", "all")
plt.tight_layout()
plt.savefig("./agreement.pdf")


# %% Annotation unification
#
# We note that annotators disagree in 2.44% of examples
#
# Possibilities:
# 1. soft labels
# 2. eliminate examples in case of disagreements
# 3. pick most consensual annotators first
#
# we will pick 2. as a baseline and discuss it later
import itertools as it

unified_annotations: List[ConllAnnotation] = []
disagreements_nb = 0

for annotations in conll_annotations:

    disagreement = False
    for a1, a2 in it.combinations(annotations, 2):
        if a1.tags != a2.tags:
            disagreement = True
            disagreements_nb += 1
            break
    if disagreement:
        continue

    unified_annotations.append(
        ConllAnnotation(
            annotations[0].tokens,
            annotations[0].tags,
            None,
            None,
            annotations[0].novel,
        )
    )

disagreement_proportion = disagreements_nb / len(conll_annotations)
print(
    f"Dropped {disagreement_proportion*100:.2f}% of examples where annotators disagreed."
)


# %% post-processing
from sacremoses import MosesPunctNormalizer
from renard.utils import search_pattern


# manually some cases where annotators annotated a token partially.
# For example, in the case of the single token "Prie-Dieu", they
# annotated "Dieu" as PER, but tokenizatino considers "Prie-Dieu" as a
# single token.
to_split = [
    "Mort-Christ",
    "Tête-Christ",
    "corne-Dieu",
    "Corps-Dieu",
    "croix-Dieu",
    "Croix-Dieu",
    "Fête-Dieu",
    "Gueule-Dieu",
    "Mort-Dieu",
    "pasque-Dieu",
    "Pasque-Dieu",
    "prie-Dieu",
    "priez-Dieu",
    "sang-Dieu",
    "Sang-Dieu",
    "tête-Dieu",
    "Tête-Dieu",
    "ventre-Dieu",
    "Vendre-Dieu",
    "Barbe-Mahom",
    "mort-Mahom",
    "Pasque-Mahom",
    "Ventre-Mahom",
]


def split(
    tokens: List[str],
    tags: List[str],
    original_token: str,
    new_tokens: List[str],
    new_tags: List[str],
) -> Tuple[List[str], List[str]]:
    appearances = search_pattern(tokens, [original_token])
    for i in appearances:
        tokens = tokens[:i] + new_tokens + tokens[i + 1 :]
        tags = tags[:i] + new_tags + tags[i + 1 :]
    assert len(tokens) == len(tags)
    return tokens, tags


for annotation in tqdm(unified_annotations, desc="fixing special cases"):
    for token_to_split in to_split:
        new_tokens, new_tags = split(
            annotation.tokens,
            annotation.tags,
            token_to_split,
            re.split(r"(-)", token_to_split),
            ["O", "O", "B-PER"],
        )
        annotation.tokens = new_tokens
        annotation.tags = new_tags


# treat the rare case of wrongly present hyphens ­
# some words are wrongly split, for example:
#
# présen O
# ­ O
# terait-elle O
#
# we remove the hyphen and glue them together again:
#
# présenterait-elle O
for annotation in tqdm(unified_annotations, desc="removing hyphens"):
    while True:
        try:
            hyphen_i = annotation.tokens.index("­")
        except ValueError:
            break
        prev_token = annotation.tokens[hyphen_i - 1]
        next_token = annotation.tokens[hyphen_i + 1]
        annotation.tokens[hyphen_i - 1] = prev_token + next_token
        annotation.tokens = (
            annotation.tokens[:hyphen_i] + annotation.tokens[hyphen_i + 2 :]
        )
        annotation.tags = annotation.tags[:hyphen_i] + annotation.tags[hyphen_i + 2 :]
        assert len(annotation.tokens) == len(annotation.tags)


# general normalization
punct_normalizer = MosesPunctNormalizer("fr", pre_replace_unicode_punct=True)
for annotation in tqdm(unified_annotations, desc="normalizing"):
    for i, token in enumerate(annotation.tokens):
        token = punct_normalizer.normalize(token)
        token = re.sub(r"…", "...", token)
        token = re.sub(r"œ", "oe", token)
        token = re.sub(r"æ", "ae", token)
        annotation.tokens[i] = token


# %% Write output conll files
for novel in startid_to_novel.values():
    path = f"./{novel}.conll"
    print(f"writing to {path}...", end="")
    with open(path, "w") as f:
        annotations = [u for u in unified_annotations if u.novel == novel]
        for annotation in annotations:
            for token, tag in zip(annotation.tokens, annotation.tags):
                f.write(f"{token} {tag}\n")
            f.write("\n")
    print("done!")
