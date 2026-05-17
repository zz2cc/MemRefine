"""Entity extraction and F1 scoring for memory quality evaluation."""

import spacy
import numpy as np
from typing import List, Set, Optional
from config import config


class EntityExtractor:
    """Extract named entities from dialogue and memory documents.

    Uses spaCy NER by default; also supports explicit field-list extraction via regex.
    """

    def __init__(self, spacy_model: str = None):
        spacy_model = spacy_model or config.spacy_model
        print(f"[NER] Loading spaCy model '{spacy_model}' ...")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"    Model '{spacy_model}' not found.")
            print(f"    Run: python -m spacy download {spacy_model}")
            print(f"    Entity extraction disabled — scores will be 0.")
            self.nlp = None

        # Entity types we care about (people, orgs, locations, dates, numbers, products)
        self.relevant_types = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
                                "MONEY", "PERCENT", "PRODUCT", "EVENT", "FAC",
                                "CARDINAL", "QUANTITY", "ORDINAL"}

    def extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text, returning set of normalized entity strings."""
        if self.nlp is None:
            return set()
        doc = self.nlp(text[:100000])  # truncate for safety
        entities = set()
        for ent in doc.ents:
            if ent.label_ in self.relevant_types:
                # Normalize: lowercase, strip whitespace
                entities.add(ent.text.lower().strip())
        return entities

    def extract_entities_by_type(self, text: str) -> dict:
        """Extract entities grouped by NER type."""
        if self.nlp is None:
            return {}
        doc = self.nlp(text[:100000])
        grouped = {}
        for ent in doc.ents:
            if ent.label_ in self.relevant_types:
                grouped.setdefault(ent.label_, set()).add(ent.text.lower().strip())
        return grouped


def compute_entity_f1(reference_entities: Set[str], candidate_entities: Set[str]) -> float:
    """Compute entity-level F1 score.

    Uses exact string match (case-insensitive, normalized).
    """
    if not reference_entities and not candidate_entities:
        return 1.0
    if not reference_entities or not candidate_entities:
        return 0.0

    ref = set(reference_entities)
    cand = set(candidate_entities)

    tp = len(ref & cand)
    fp = len(cand - ref)
    fn = len(ref - cand)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1


def compute_entity_jaccard(ref: Set[str], cand: Set[str]) -> float:
    """Jaccard similarity over entity sets (alternative to F1)."""
    if not ref and not cand:
        return 1.0
    intersection = len(ref & cand)
    union = len(ref | cand)
    return intersection / max(union, 1)


def compute_entity_precision(ref: Set[str], cand: Set[str]) -> float:
    tp = len(ref & cand)
    fp = len(cand - ref)
    return tp / max(tp + fp, 1)


def compute_entity_recall(ref: Set[str], cand: Set[str]) -> float:
    tp = len(ref & cand)
    fn = len(ref - cand)
    return tp / max(tp + fn, 1)
