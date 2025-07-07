from typing import cast
import glob, os
from renard.ner_utils import load_conll2002_bio
from datasets import Dataset, DatasetInfo, load_dataset, concatenate_datasets

# general info
bibtex_citation = """@InProceedings{Maurel2025,
  authors = {Maurel, P. and Amalvy, A. and Labatut, V. and Alrahabi, M.},
  title = {Du repérage à l’analyse : un modèle pour la reconnaissance d’entités nommées dans les textes littéraires en français},
  booktitle = {Digital Humanities 2025},
  year = {2025},
}"""
info = DatasetInfo(
    description="A dataset of 7 French novels.",
    citation=bibtex_citation,
    homepage="https://github.com/obtic-sorbonne/7-romans",
    license="mit",
)


# alias-resolution subset
dataset = load_dataset("csv", data_files=glob.glob("./alias-resolution/*.csv"))
dataset = dataset.remove_columns("Metadata")

dataset = []
for novel in glob.glob("./alias-resolution/*.csv"):
    novel_dataset = cast(Dataset, load_dataset("csv", data_files=novel)["train"])  # type: ignore
    novel_dataset = novel_dataset.remove_columns("Metadata")
    for col in novel_dataset.column_names:
        novel_dataset = novel_dataset.rename_column(col, col.lower())
    novel_name = os.path.splitext(os.path.basename(novel))[0]
    novel_dataset = novel_dataset.add_column(
        name="novel", column=[novel_name for _ in range(len(novel_dataset))]
    )  # type: ignore
    dataset.append(novel_dataset)
dataset = concatenate_datasets(dataset, info=info)
dataset.push_to_hub("compnet-renard/7-romans-alias-resolution", "alias-resolution")


# text subset
text_dataset = []
for novel in glob.glob("./ner/*.conll"):
    novel_name = os.path.splitext(os.path.basename(novel))[0]
    _, tokens, _ = load_conll2002_bio(novel, separator=" ")
    novel_dataset = Dataset.from_dict({"tokens": [tokens], "novel": [novel_name]})
    text_dataset.append(novel_dataset)
text_dataset = concatenate_datasets(text_dataset, info=info)
text_dataset.push_to_hub("compnet-renard/7-romans-alias-resolution", "text")
