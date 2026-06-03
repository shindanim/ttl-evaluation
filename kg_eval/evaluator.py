from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, URIRef


DEFAULT_COSINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


def _greedy_matches(
    candidates: list[tuple[float, int, int, TripleRecord, TripleRecord]],
    true_triples: list[TripleRecord],
    pred_triples: list[TripleRecord],
) -> tuple[list[dict], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
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
                "score": round(float(score), 4),
            }
        )

    false_positives = [
        triple.as_tuple() for index, triple in enumerate(pred_triples) if index not in used_pred
    ]
    false_negatives = [
        triple.as_tuple() for index, triple in enumerate(true_triples) if index not in used_true
    ]
    return matches, false_positives, false_negatives


def evaluate_cosine(
    true_ttl: str | Path,
    pred_ttl: str | Path,
    threshold: float = 0.85,
    model_name: str = DEFAULT_COSINE_MODEL,
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

    matches, false_positives, false_negatives = _greedy_matches(
        candidates, true_triples, pred_triples
    )

    return {
        "mode": "cosine",
        "threshold": threshold,
        "model": model_name,
        **_metrics(len(matches), len(pred_triples), len(true_triples)),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_cross_encoder(
    true_ttl: str | Path,
    pred_ttl: str | Path,
    threshold: float = 0.0,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    batch_size: int = 32,
) -> dict:
    from sentence_transformers import CrossEncoder

    true_triples = load_triples(true_ttl)
    pred_triples = load_triples(pred_ttl)

    if not true_triples or not pred_triples:
        return {
            "mode": "cross",
            "threshold": threshold,
            "model": model_name,
            **_metrics(0, len(pred_triples), len(true_triples)),
            "matches": [],
            "false_positives": [triple.as_tuple() for triple in pred_triples],
            "false_negatives": [triple.as_tuple() for triple in true_triples],
        }

    pairs = []
    indexes = []
    for true_index, true_triple in enumerate(true_triples):
        for pred_index, pred_triple in enumerate(pred_triples):
            pairs.append((true_triple.text(), pred_triple.text()))
            indexes.append((true_index, pred_index, true_triple, pred_triple))

    model = CrossEncoder(model_name)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    candidates = []
    for score, (true_index, pred_index, true_triple, pred_triple) in zip(scores, indexes):
        score_value = float(score)
        if score_value >= threshold:
            candidates.append((score_value, true_index, pred_index, true_triple, pred_triple))

    matches, false_positives, false_negatives = _greedy_matches(
        candidates, true_triples, pred_triples
    )

    return {
        "mode": "cross",
        "threshold": threshold,
        "model": model_name,
        **_metrics(len(matches), len(pred_triples), len(true_triples)),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def default_model_for_mode(mode: str) -> str:
    if mode == "cross":
        return DEFAULT_CROSS_ENCODER_MODEL
    return DEFAULT_COSINE_MODEL


def evaluate(
    true_ttl: str | Path,
    pred_ttl: str | Path,
    mode: str,
    threshold: float,
    model_name: str | None,
) -> dict:
    if mode == "strict":
        return evaluate_strict(true_ttl, pred_ttl)
    model = model_name or default_model_for_mode(mode)
    if mode == "cosine":
        return evaluate_cosine(true_ttl, pred_ttl, threshold, model)
    if mode == "cross":
        return evaluate_cross_encoder(true_ttl, pred_ttl, threshold, model)
    raise ValueError(f"Unsupported mode: {mode}")


def _print_summary(result: dict) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate two Turtle knowledge graphs.")
    parser.add_argument("--true", required=True, dest="true_ttl", help="Path to gold KG TTL.")
    parser.add_argument("--pred", required=True, dest="pred_ttl", help="Path to predicted KG TTL.")
    parser.add_argument("--mode", choices=["strict", "cosine", "cross"], default="strict")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Match threshold. Defaults to 0.85 for cosine and 0.0 for cross.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name used in cosine/cross mode. Defaults depend on mode.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    threshold = args.threshold
    if threshold is None:
        threshold = 0.0 if args.mode == "cross" else 0.85
    result = evaluate(args.true_ttl, args.pred_ttl, args.mode, threshold, args.model)
    _print_summary(result)


if __name__ == "__main__":
    main()
