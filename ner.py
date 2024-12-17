from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class NEREntity:

    tokens: List[str]
    start_idx: int
    end_idx: int

    #: NER class (without BIO prefix as in ``PER`` and not ``B-PER``)
    tag: str

    def shifted(self, shift: int) -> NEREntity:
        self_dict = vars(self)
        self_dict["start_idx"] = self.start_idx + shift
        self_dict["end_idx"] = self.end_idx + shift
        return self.__class__(**self_dict)

    def __eq__(self, other: NEREntity) -> bool:
        return (
            self.tokens == other.tokens
            and self.start_idx == other.start_idx
            and self.end_idx == other.end_idx
        )

    def __hash__(self) -> int:
        return hash(tuple(self.tokens) + (self.start_idx, self.end_idx))


def entities_to_BIO(tokens: List[str], entities: List[NEREntity]) -> List[str]:
    """Convert a list of entities to BIO tags."""
    tags = ["O"] * len(tokens)
    for entity in entities:
        entity_len = entity.end_idx - entity.start_idx
        tags[entity.start_idx : entity.end_idx] = [f"B-{entity.tag}"] + [
            f"I-{entity.tag}"
        ] * (entity_len - 1)
    return tags
