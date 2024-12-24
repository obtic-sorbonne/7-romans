from __future__ import annotations
from typing import List, Dict, Literal, Optional
import functools as ft
from sacred import Experiment
from sacred.commands import print_config
from sacred.run import Run
from datasets import Dataset, concatenate_datasets
from transformers import (
    PreTrainedModel,
    Trainer,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from renard.ner_utils import hgdataset_from_conll2002, _tokenize_and_align_labels
from data import NER_ID2LABEL, NovelTitle


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


@ex.automain
def main(
    _run: Run,
    model_id: str,
    output_dir: Optional[str],
    hg_training_kwargs: Dict,
):
    print_config(_run)

    novels: List[Literal[NovelTitle]] = [
        "BelAmi",
        "EugénieGrandet",
        "Germinal",
        "LeRougeEtLeNoir",
        "LesTroisMousquetaires",
        "MadameBovary",
        "NotreDameDeParis",
    ]
    # [(novel title, (ner_dataset, characters)) ...]
    datasets = [
        hgdataset_from_conll2002(f"./ner/{name}.conll", separator=" ")
        for name in novels
    ]
    train = concatenate_datasets(datasets)
    targs = TrainingArguments(**hg_training_kwargs)
    model = train_ner_model(_run, model_id, train, train, targs)

    if not output_dir is None:
        model.save_pretrained(f"{output_dir}/{model_id}-literary-NER")
