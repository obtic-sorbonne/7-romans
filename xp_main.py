from __future__ import annotations
from typing import List, Dict, Literal, Optional
import os
import functools as ft
from statistics import mean
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from datasets import Dataset, concatenate_datasets
from more_itertools import flatten
from transformers import (
    PreTrainedModel,
    Trainer,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from renard.ner_utils import (
    hgdataset_from_conll2002,
    _tokenize_and_align_labels,
    ner_entities,
)
from renard.pipeline.core import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from data import NER_ID2LABEL, NovelTitle, load_novel

from utils import archive_pipeline_state_
from measures import (
    align_characters,
    score_ner,
    score_network_extraction_edges,
    score_character_unification,
)


class SacredTrainer(Trainer):
    def __init__(self, _run: Run, **kwargs):
        super().__init__(**kwargs)
        self._run = _run

    def evaluate(self, **kwargs) -> Dict[str, float]:
        metrics = super().evaluate(**kwargs)
        for k, v in metrics.items():
            self._run.log_scalar(k, v)
        return metrics

    def log(self, logs: Dict[str, float]):
        super().log(logs)
        if "loss" in logs:
            self._run.log_scalar("loss", logs["loss"])
        if "learning_rate" in logs:
            self._run.log_scalar("learning_rate", logs["learning_rate"])


ex = Experiment("train")


def train_ner_model(
    _run: Run, hg_id: str, train: Dataset, test: Dataset, targs: TrainingArguments
) -> PreTrainedModel:
    # BERT tokenizer splits tokens into subtokens. The
    # tokenize_and_align_labels function correctly aligns labels and
    # subtokens.
    if "camembert" in hg_id:
        tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained(hg_id)
    train = train.map(
        ft.partial(_tokenize_and_align_labels, tokenizer=tokenizer), batched=True
    )
    test = test.map(
        ft.partial(_tokenize_and_align_labels, tokenizer=tokenizer), batched=True
    )

    label_lst = train.features["labels"].feature.names
    model = AutoModelForTokenClassification.from_pretrained(
        hg_id,
        num_labels=len(label_lst),
        id2label=NER_ID2LABEL,
        label2id={v: k for k, v in NER_ID2LABEL.items()},
    )

    trainer = SacredTrainer(
        _run,
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=train,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()

    return model


@ex.config
def config():
    model_id: str
    output_dir: Optional[str] = None
    # see https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/trainer#transformers.TrainingArguments
    hg_training_kwargs: Dict
    # in tokens
    co_occurrences_dist: int


@ex.automain
def main(
    _run: Run,
    model_id: str,
    output_dir: Optional[str],
    hg_training_kwargs: Dict,
    co_occurrences_dist: int,
):
    print_config(_run)

    novels: List[Literal[NovelTitle]] = [
        "BelAmi",
        "EugénieGrandet",
        "Germinal",
        "LeRougeEtLeNoir",
        "LesTroisMousquetaires",
        "NotreDameDeParis",
    ]
    # [(novel title, (ner_dataset, characters)) ...]
    datasets = [(name, load_novel(name)) for name in novels]

    gold_pipeline = Pipeline(
        [
            CoOccurrencesGraphExtractor(
                co_occurrences_dist=(co_occurrences_dist, "tokens")
            ),
        ]
    )

    precision_lst = []
    recall_lst = []
    f1_lst = []

    for test_title, dataset in datasets:
        ner_test, gold_characters = dataset
        print(f"testing on {test_title}")
        ner_train = [dataset[0] for title, dataset in datasets if title != test_title]
        ner_train = concatenate_datasets(ner_train)

        # train NER model
        targs = TrainingArguments(**hg_training_kwargs)
        model = train_ner_model(_run, model_id, ner_train, ner_test, targs)

        # gold pipeline run
        tokens = list(flatten(ner_test["tokens"]))
        gold_tags = [NER_ID2LABEL[l] for l in flatten(ner_test["labels"])]
        gold_entities = ner_entities(tokens, gold_tags)
        out_gold = gold_pipeline(
            tokens=tokens,
            sentences=ner_test["tokens"],
            entities=gold_entities,
            characters=gold_characters,
        )
        archive_pipeline_state_(_run, out_gold, f"{test_title}_state_gold")

        # pipeline run
        if "camembert" in model_id:
            tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipeline = Pipeline(
            [
                BertNamedEntityRecognizer(model=model, tokenizer=tokenizer),
                GraphRulesCharacterUnifier(),
                CoOccurrencesGraphExtractor(
                    co_occurrences_dist=(co_occurrences_dist, "tokens")
                ),
            ],
            progress_report=None,
            lang="fra",
        )

        out = pipeline(tokens=tokens, sentences=ner_test["tokens"])
        archive_pipeline_state_(_run, out, f"{test_title}_state")

        # Network extraction scoring
        assert not out.characters is None
        assert not out_gold.characters is None
        vertex_precision, vertex_recall, vertex_f1 = score_character_unification(
            [character.names for character in out_gold.characters],
            [character.names for character in out.characters],
        )
        _run.log_scalar(f"{test_title}.VertexPrecision", vertex_precision)
        _run.log_scalar(f"{test_title}.VertexRecall", vertex_recall)
        _run.log_scalar(f"{test_title}.VertexF1", vertex_f1)

        mapping, _ = align_characters(out_gold.characters, out.characters)
        edge_precision, edge_recall, edge_f1 = score_network_extraction_edges(
            out_gold.character_network, out.character_network, mapping
        )
        _run.log_scalar(f"{test_title}.EdgePrecision", edge_precision)
        _run.log_scalar(f"{test_title}.EdgeRecall", edge_recall)
        _run.log_scalar(f"{test_title}.EdgeF1", edge_f1)

        w_edge_precision, w_edge_recall, w_edge_f1 = score_network_extraction_edges(
            out_gold.character_network, out.character_network, mapping, weighted=True
        )
        _run.log_scalar(f"{test_title}.WeightedEdgePrecision", w_edge_precision)
        _run.log_scalar(f"{test_title}.WeightedEdgeRecall", w_edge_recall)
        _run.log_scalar(f"{test_title}.WeightedEdgeF1", w_edge_f1)

        # NER scoring
        assert not out.entities is None
        assert not out_gold.entities is None
        precision, recall, f1 = score_ner(
            tokens, out.entities, out_gold.entities, ignore_classes=None
        )
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)
        _run.log_scalar(f"{test_title}.Precision", precision)
        _run.log_scalar(f"{test_title}.Recall", recall)
        _run.log_scalar(f"{test_title}.F1", f1)

        if not output_dir is None:
            model.save_pretrained(os.path.join(output_dir, test_title))

    mean_precision = mean(precision_lst)
    mean_recall = mean(recall_lst)
    mean_f1 = mean(f1_lst)
    print(f"{mean_precision}")
    print(f"{mean_recall}")
    print(f"{mean_f1}")
    _run.log_scalar("Mean Precision", mean_precision)
    _run.log_scalar("Mean Recall", mean_recall)
    _run.log_scalar("Mean F1", mean_f1)
