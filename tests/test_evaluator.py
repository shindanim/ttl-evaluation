from kg_eval.evaluator import DEFAULT_CROSS_ENCODER_MODEL, default_model_for_mode, evaluate_strict


def test_evaluate_strict_examples(tmp_path):
    true_ttl = tmp_path / "true.ttl"
    pred_ttl = tmp_path / "pred.ttl"
    true_ttl.write_text(
        """
        @prefix ex: <http://example.org/> .
        ex:Alice ex:knows ex:Bob .
        ex:Alice ex:worksAt ex:Acme .
        ex:Bob ex:livesIn ex:Seoul .
        ex:Acme ex:locatedIn ex:Seoul .
        """,
        encoding="utf-8",
    )
    pred_ttl.write_text(
        """
        @prefix ex: <http://example.org/> .
        ex:Alice ex:knows ex:Bob .
        ex:Alice ex:employedBy ex:AcmeCorp .
        ex:Bob ex:livesIn ex:Seoul .
        ex:AcmeCorp ex:locatedIn ex:Seoul .
        """,
        encoding="utf-8",
    )

    result = evaluate_strict(true_ttl, pred_ttl)

    assert result["true_positives"] == 2
    assert result["predicted_count"] == 4
    assert result["gold_count"] == 4
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_cross_mode_default_model():
    assert default_model_for_mode("cross") == DEFAULT_CROSS_ENCODER_MODEL
