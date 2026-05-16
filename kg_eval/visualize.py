from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from pyvis.network import Network

from kg_eval.evaluator import evaluate, load_triples


COLORS = {
    "default_true": "#4E79A7",
    "default_pred": "#F28E2B",
    "matched": "#E15759",
    "literal": "#B07AA1",
}


def matched_nodes(eval_result: dict) -> set[str]:
    nodes: set[str] = set()
    for match in eval_result.get("matches", []):
        for triple_key in ("true", "pred"):
            subject, _, obj = match[triple_key]
            nodes.add(subject)
            nodes.add(obj)
    return nodes


def graph_payload(ttl_path: str | Path, namespace: str, matched: set[str]) -> dict:
    triples = load_triples(ttl_path)
    nodes = {}
    edges = []
    for subject, predicate, obj in [triple.as_tuple() for triple in triples]:
        for node_id in (subject, obj):
            key = f"{namespace}:{node_id}"
            if key not in nodes:
                is_matched = node_id in matched
                nodes[key] = {
                    "id": key,
                    "label": node_id,
                    "group": "matched" if is_matched else namespace,
                    "matched": is_matched,
                }
        edges.append(
            {
                "from": f"{namespace}:{subject}",
                "to": f"{namespace}:{obj}",
                "label": predicate,
                "arrows": "to",
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


def make_network(true_ttl: str | Path, pred_ttl: str | Path, eval_result: dict) -> str:
    matched = matched_nodes(eval_result)
    true_payload = graph_payload(true_ttl, "true", matched)
    pred_payload = graph_payload(pred_ttl, "pred", matched)
    payload = {
        "nodes": true_payload["nodes"] + pred_payload["nodes"],
        "edges": true_payload["edges"] + pred_payload["edges"],
        "result": eval_result,
    }

    net = Network(height="720px", width="100%", directed=True, bgcolor="#fbfbf8")
    net.set_options(
        """
        {
          "nodes": {"shape": "dot", "font": {"size": 16}},
          "edges": {"font": {"size": 12, "align": "middle"}, "smooth": true},
          "physics": {"stabilization": true, "barnesHut": {"springLength": 150}}
        }
        """
    )

    for node in payload["nodes"]:
        group = node["group"]
        color = COLORS["matched"] if group == "matched" else COLORS[f"default_{group}"]
        net.add_node(
            node["id"],
            label=node["label"],
            color=color,
            title=f"{group}: {node['label']}",
            group=group,
            matched=node["matched"],
        )
    for edge in payload["edges"]:
        net.add_edge(edge["from"], edge["to"], label=edge["label"], arrows="to")

    with NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        net.write_html(tmp.name, notebook=False)
        html = Path(tmp.name).read_text(encoding="utf-8")

    matched_items = []
    for index, match in enumerate(eval_result.get("matches", []), start=1):
        true_triple = " | ".join(match["true"])
        pred_triple = " | ".join(match["pred"])
        matched_items.append(
            f"""
            <li>
              <div class="match-row-title">#{index} score={match["score"]}</div>
              <div><strong>true</strong>: {true_triple}</div>
              <div><strong>pred</strong>: {pred_triple}</div>
            </li>
            """
        )
    matched_list = "\n".join(matched_items) or "<li>No matched triples.</li>"

    threshold_value = eval_result.get("threshold", 0.85)
    controls = f"""
    <style>
      body {{ margin: 0; font-family: Inter, Segoe UI, sans-serif; color: #242424; }}
      .kg-toolbar {{
        display: flex; align-items: center; gap: 12px; padding: 12px 16px;
        flex-wrap: wrap;
        border-bottom: 1px solid #ddd; background: #ffffff;
      }}
      .kg-toolbar button {{
        border: 1px solid #b7b7b7; background: #fff; padding: 8px 12px;
        border-radius: 6px; cursor: pointer; font-weight: 600;
      }}
      .kg-toolbar button.active {{ background: #242424; color: #fff; }}
      .kg-toolbar input {{
        width: 76px; padding: 7px 8px; border: 1px solid #b7b7b7; border-radius: 6px;
      }}
      .mode-group {{
        display: inline-flex; gap: 6px; align-items: center; padding-right: 4px;
      }}
      .metric {{ font-size: 14px; color: #555; }}
      .match-panel {{
        display: none; max-height: 260px; overflow: auto; padding: 12px 16px;
        border-bottom: 1px solid #ddd; background: #fffdf8;
      }}
      .match-panel.open {{ display: block; }}
      .match-panel ol {{ margin: 0; padding-left: 24px; }}
      .match-panel li {{ margin-bottom: 10px; font-size: 13px; line-height: 1.45; }}
      .match-row-title {{ font-weight: 700; color: #9c2628; }}
    </style>
    <div class="kg-toolbar">
      <button class="view-filter active" data-view="all">All</button>
      <button class="view-filter" data-view="true">true.ttl</button>
      <button class="view-filter" data-view="pred">pred.ttl</button>
      <span class="mode-group">
        <button class="mode-button" data-mode="strict">Strict</button>
        <button class="mode-button" data-mode="cosine">Cosine</button>
        <input id="thresholdInput" type="number" min="0" max="1" step="0.01" value="{threshold_value}">
      </span>
      <button id="toggleMatched" class="active">Highlight matched nodes</button>
      <button id="toggleMatchedTriples">Matched triples</button>
      <button id="fitGraph">Fit graph</button>
      <span class="metric">Mode: {eval_result["mode"]}</span>
      <span class="metric">Precision: {eval_result["precision"]:.3f}</span>
      <span class="metric">Recall: {eval_result["recall"]:.3f}</span>
      <span class="metric">F1: {eval_result["f1"]:.3f}</span>
    </div>
    <div id="matchedTriplesPanel" class="match-panel">
      <ol>
        {matched_list}
      </ol>
    </div>
    <script>
      const kgPayload = {json.dumps(payload, ensure_ascii=False)};
      const normalColors = {{ true: "{COLORS["default_true"]}", pred: "{COLORS["default_pred"]}", matched: "{COLORS["matched"]}" }};
      const currentMode = "{eval_result["mode"]}";
      let highlighted = true;
      let currentView = "all";

      function nodeNamespace(node) {{
        return node.id.split(":")[0];
      }}

      function edgeNamespace(edge) {{
        return edge.from.split(":")[0];
      }}

      function visibleNodes() {{
        return kgPayload.nodes.filter((node) => currentView === "all" || nodeNamespace(node) === currentView);
      }}

      function visibleEdges() {{
        return kgPayload.edges.filter((edge) => currentView === "all" || edgeNamespace(edge) === currentView);
      }}

      function colorizedNode(node) {{
          const ns = node.id.split(":")[0];
          return {{
            ...node,
            color: node.matched && highlighted ? normalColors.matched : normalColors[ns]
          }};
      }}

      function renderView() {{
        nodes.clear();
        edges.clear();
        nodes.add(visibleNodes().map(colorizedNode));
        edges.add(visibleEdges());
        network.fit({{ animation: {{ duration: 300, easingFunction: "easeInOutQuad" }} }});
      }}

      document.addEventListener("DOMContentLoaded", () => {{
        document.querySelectorAll(".mode-button").forEach((button) => {{
          button.classList.toggle("active", button.dataset.mode === currentMode);
          button.addEventListener("click", (event) => {{
            const mode = event.currentTarget.dataset.mode;
            const params = new URLSearchParams(window.location.search);
            params.set("mode", mode);
            if (mode === "cosine") {{
              params.set("threshold", document.getElementById("thresholdInput").value || "0.85");
            }} else {{
              params.delete("threshold");
            }}
            window.location.search = params.toString();
          }});
        }});
        document.querySelectorAll(".view-filter").forEach((button) => {{
          button.addEventListener("click", (event) => {{
            currentView = event.currentTarget.dataset.view;
            document.querySelectorAll(".view-filter").forEach((item) => {{
              item.classList.toggle("active", item.dataset.view === currentView);
            }});
            renderView();
          }});
        }});
        document.getElementById("toggleMatched").addEventListener("click", (event) => {{
          highlighted = !highlighted;
          event.currentTarget.classList.toggle("active", highlighted);
          renderView();
        }});
        document.getElementById("toggleMatchedTriples").addEventListener("click", (event) => {{
          const panel = document.getElementById("matchedTriplesPanel");
          const isOpen = panel.classList.toggle("open");
          event.currentTarget.classList.toggle("active", isOpen);
        }});
        document.getElementById("fitGraph").addEventListener("click", () => network.fit());
        renderView();
      }});
    </script>
    """
    return html.replace("<body>", f"<body>{controls}", 1)


def write_html(
    true_ttl: str | Path,
    pred_ttl: str | Path,
    output_path: str | Path,
    mode: str = "strict",
    threshold: float = 0.85,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Path:
    eval_result = evaluate(true_ttl, pred_ttl, mode, threshold, model_name)
    html = make_network(true_ttl, pred_ttl, eval_result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render TTL KG visualization as HTML.")
    parser.add_argument("--true", required=True, dest="true_ttl")
    parser.add_argument("--pred", required=True, dest="pred_ttl")
    parser.add_argument("--output", default="kg_visualization.html")
    parser.add_argument("--mode", choices=["strict", "cosine"], default="strict")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = write_html(
        args.true_ttl,
        args.pred_ttl,
        args.output,
        args.mode,
        args.threshold,
        args.model,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
