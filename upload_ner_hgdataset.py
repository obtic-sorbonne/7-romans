import glob, os
from renard.ner_utils import hgdataset_from_conll2002
from datasets import DatasetInfo, concatenate_datasets


dataset = []
for novel in glob.glob("./ner/*.conll"):
    novel_dataset = hgdataset_from_conll2002(novel, separator=" ")
    novel_name = os.path.splitext(os.path.basename(novel))[0]
    novel_dataset = novel_dataset.add_column(
        name="novel", column=[novel_name for _ in range(len(novel_dataset))]
    )  # type: ignore
    dataset.append(novel_dataset)

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
dataset = concatenate_datasets(dataset, info=info)
dataset.push_to_hub("compnet-renard/7-romans-ner")
