from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, URIRef


@dataclass(frozen=True)
class TripleRecord:
    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


def compact_term(term) -> str:
    if isinstance(term, URIRef):
        text = str(term)
        for sep in ("#", "/"):
            if sep in text:
                tail = text.rstrip(sep).split(sep)[-1]
                if tail:
                    return tail
        return text
    if isinstance(term, Literal):
        return str(term)
    return str(term)


def load_triples(ttl_path: str | Path) -> list[TripleRecord]:
    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")
    triples = [
        TripleRecord(compact_term(s), compact_term(p), compact_term(o))
        for s, p, o in graph
    ]
    return sorted(triples, key=lambda item: item.as_tuple())


def _metrics(tp: int, pred_count: int, true_count: int) -> dict[str, float | int]:
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / true_count if true_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "predicted_count": pred_count,
        "gold_count": true_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_strict(true_ttl: str | Path, pred_ttl: str | Path) -> dict:
    true_triples = load_triples(true_ttl)
    pred_triples = load_triples(pred_ttl)
    true_set = {triple.as_tuple() for triple in true_triples}
    pred_set = {triple.as_tuple() for triple in pred_triples}
    matched = sorted(true_set & pred_set)
    false_positives = sorted(pred_set - true_set)
    false_negatives = sorted(true_set - pred_set)

    return {
        "mode": "strict",
        **_metrics(len(matched), len(pred_set), len(true_set)),
        "matches": [
            {"true": item, "pred": item, "score": 1.0}
            for item in matched
        ],
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _cosine_matrix(left_texts: list[str], right_texts: list[str], model_name: str):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)
    left_embeddings = model.encode(left_texts, normalize_embeddings=True)
    right_embeddings = model.encode(right_texts, normalize_embeddings=True)
    return cosine_similarity(left_embeddings, right_embeddings)


def _triple_score(matrix, true_index: int, pred_index: int) -> float:
    base = true_index * 3
    cand = pred_index * 3
    subject = matrix[base][cand]
    predicate = matrix[base + 1][cand + 1]
    obj = matrix[base + 2][cand + 2]
    return float((subject + predicate + obj) / 3)


def evaluate_cosine(
    true_ttl: str | Path,
    pred_ttl: str | Path,
    threshold: float = 0.85,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    true_triples = load_triples(true_ttl)
    pred_triples = load_triples(pred_ttl)
    true_terms = [term for triple in true_triples for term in triple.as_tuple()]
    pred_terms = [term for triple in pred_triples for term in triple.as_tuple()]

    if not true_triples or not pred_triples:
        return {
            "mode": "cosine",
            "threshold": threshold,
            **_metrics(0, len(pred_triples), len(true_triples)),
            "matches": [],
            "false_positives": [triple.as_tuple() for triple in pred_triples],
            "false_negatives": [triple.as_tuple() for triple in true_triples],
        }

    matrix = _cosine_matrix(true_terms, pred_terms, model_name)
    candidates = []
    for true_index, true_triple in enumerate(true_triples):
        for pred_index, pred_triple in enumerate(pred_triples):
            score = _triple_score(matrix, true_index, pred_index)
            if score >= threshold:
                candidates.append((score, true_index, pred_index, true_triple, pred_triple))

    candidates.sort(reverse=True, key=lambda item: item[0])
    used_true: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for score, true_index, pred_index, true_triple, pred_triple in candidates:
        if true_index in used_true or pred_index in used_pred:
            continue
        used_true.add(true_index)
        used_pred.add(pred_index)
        matches.append(
            {
                "true": true_triple.as_tuple(),
                "pred": pred_triple.as_tuple(),
                "score": round(score, 4),
            }
        )

    false_positives = [
        triple.as_tuple() for index, triple in enumerate(pred_triples) if index not in used_pred
    ]
    false_negatives = [
        triple.as_tuple() for index, triple in enumerate(true_triples) if index not in used_true
    ]

    return {
        "mode": "cosine",
        "threshold": threshold,
        "model": model_name,
        **_metrics(len(matches), len(pred_triples), len(true_triples)),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate(
    true_ttl: str | Path,
    pred_ttl: str | Path,
    mode: str,
    threshold: float,
    model_name: str,
) -> dict:
    if mode == "strict":
        return evaluate_strict(true_ttl, pred_ttl)
    if mode == "cosine":
        return evaluate_cosine(true_ttl, pred_ttl, threshold, model_name)
    raise ValueError(f"Unsupported mode: {mode}")


def _print_summary(result: dict) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate two Turtle knowledge graphs.")
    parser.add_argument("--true", required=True, dest="true_ttl", help="Path to gold KG TTL.")
    parser.add_argument("--pred", required=True, dest="pred_ttl", help="Path to predicted KG TTL.")
    parser.add_argument("--mode", choices=["strict", "cosine"], default="strict")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model used in cosine mode.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate(args.true_ttl, args.pred_ttl, args.mode, args.threshold, args.model)
    _print_summary(result)


if __name__ == "__main__":
    main()

