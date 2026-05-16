from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request

from kg_eval.evaluator import evaluate
from kg_eval.visualize import make_network


def create_app(true_ttl: str | Path, pred_ttl: str | Path) -> Flask:
    app = Flask(__name__)
    true_path = Path(true_ttl)
    pred_path = Path(pred_ttl)

    @app.get("/")
    def index():
        mode = request.args.get("mode", "strict")
        threshold = float(request.args.get("threshold", "0.85"))
        model = request.args.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        result = evaluate(true_path, pred_path, mode, threshold, model)
        return make_network(true_path, pred_path, result)

    @app.get("/api/evaluate")
    def api_evaluate():
        mode = request.args.get("mode", "strict")
        threshold = float(request.args.get("threshold", "0.85"))
        model = request.args.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        return jsonify(evaluate(true_path, pred_path, mode, threshold, model))

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve KG evaluation visualization.")
    parser.add_argument("--true", required=True, dest="true_ttl")
    parser.add_argument("--pred", required=True, dest="pred_ttl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(args.true_ttl, args.pred_ttl)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()

