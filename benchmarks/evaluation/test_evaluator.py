import pytest

from .evaluator import (
    EvalEntity,
    EvalTriple,
    EvalGraph,
    normalize_text,
    evaluate_graph,
    GraphEvaluator,
)


def make_triple(s: str, p: str, o: str, p_uri: str = None) -> EvalTriple:
    """Helper function to quickly create an EvalTriple for testing."""
    return EvalTriple(
        subject=EvalEntity(surface_form=s),
        predicate=EvalEntity(surface_form=p, uri=p_uri),
        object=EvalEntity(surface_form=o),
    )


def test_normalize_text():
    # Test empty
    assert normalize_text(None) == ""
    assert normalize_text("") == ""

    # Test basic lowercase and strip
    assert normalize_text("  HeLlO  ") == "hello"

    # Test underscores to spaces
    assert normalize_text("Albert_Einstein") == "albert einstein"

    # Test HTML unescaping
    assert normalize_text("Dunn &amp; Bradstreet") == "dunn & bradstreet"

    # Test URL unquoting
    assert normalize_text("hello%20world") == "hello world"


def test_evaluate_graph_exact_match():
    gold = EvalGraph([make_triple("Albert Einstein", "born in", "Ulm")])
    # Exact match (case insensitive due to normalization)
    gen = EvalGraph([make_triple("albert einstein", "BORN IN", "ulm")])

    res = evaluate_graph(gen, gold)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["f1_score"] == 1.0


def test_evaluate_graph_fuzzy_match():
    gold = EvalGraph([make_triple("Albert Einstein", "born in", "Ulm")])
    # Slight misspelling but should pass default 90.0 threshold
    gen = EvalGraph([make_triple("Albert Einstien", "born in", "Ulm")])

    res = evaluate_graph(gen, gold, match_threshold=90.0)
    assert res["true_positives"] == 1

    # Too big of a misspelling, should fail
    gen_fail = EvalGraph([make_triple("Albert", "born in", "Ulm")])
    res_fail = evaluate_graph(gen_fail, gold, match_threshold=90.0)
    assert res_fail["true_positives"] == 0
    assert res_fail["false_positives"] == 1
    assert res_fail["false_negatives"] == 1


def test_evaluate_graph_uri_matching():
    # Even if the text is completely different, matching URIs should result in a match for predicates
    gold = EvalGraph(
        [make_triple("Albert Einstein", "place of birth", "Ulm", p_uri="P19")]
    )
    gen = EvalGraph([make_triple("Albert Einstein", "was born at", "Ulm", p_uri="P19")])

    res = evaluate_graph(gen, gold)
    assert res["true_positives"] == 1

    # If URIs are present but mismatch, it should fail regardless of text match
    gold_mismatch = EvalGraph(
        [make_triple("Albert Einstein", "born in", "Ulm", p_uri="P19")]
    )
    gen_mismatch = EvalGraph(
        [make_triple("Albert Einstein", "born in", "Ulm", p_uri="P20")]
    )

    res_mismatch = evaluate_graph(gen_mismatch, gold_mismatch)
    assert res_mismatch["true_positives"] == 0


def test_evaluate_graph_class_fallback():
    # "is a", "type", "instance of" etc. should match each other
    gold = EvalGraph([make_triple("Albert Einstein", "is a", "Physicist")])
    gen = EvalGraph([make_triple("Albert Einstein", "type", "Physicist")])

    res = evaluate_graph(gen, gold)
    assert res["true_positives"] == 1


def test_empty_evaluator():
    evaluator = GraphEvaluator()
    res = evaluator.get_aggregated_results()
    assert res == {}


def test_aggregated_metrics():
    evaluator = GraphEvaluator(match_threshold=90.0)

    # Graph 1: 1 TP, 1 FP, 0 FN
    # Precision = 0.5, Recall = 1.0, F1 = 0.666...
    gold_1 = EvalGraph([make_triple("A", "B", "C")])
    gen_1 = EvalGraph(
        [make_triple("A", "B", "C"), make_triple("X", "Y", "Z")]  # Hallucination (FP)
    )
    res_1 = evaluator.evaluate(gen_1, gold_1)
    assert res_1["true_positives"] == 1
    assert res_1["false_positives"] == 1
    assert res_1["f1_score"] == pytest.approx(0.6666, abs=5e-4)

    # Graph 2: 2 TP, 0 FP, 0 FN
    # Precision = 1.0, Recall = 1.0, F1 = 1.0
    gold_2 = EvalGraph([make_triple("1", "2", "3"), make_triple("4", "5", "6")])
    gen_2 = EvalGraph([make_triple("1", "2", "3"), make_triple("4", "5", "6")])
    res_2 = evaluator.evaluate(gen_2, gold_2)
    assert res_2["true_positives"] == 2
    assert res_2["f1_score"] == 1.0

    # Aggregate Check
    agg = evaluator.get_aggregated_results()

    # Totals
    assert agg["total_graphs_evaluated"] == 2
    assert agg["total_true_positives"] == 3
    assert agg["total_false_positives"] == 1
    assert agg["total_false_negatives"] == 0

    # Micro Metrics
    # Total Gen = 4 (TP:3 + FP:1)
    # Total Gold = 3 (TP:3 + FN:0)
    # Micro P = 3/4 = 0.75
    # Micro R = 3/3 = 1.0
    # Micro F1 = 2 * (0.75 * 1.0) / (1.75) = 0.8571
    assert agg["micro_precision"] == 0.75
    assert agg["micro_recall"] == 1.0
    assert agg["micro_f1"] == pytest.approx(0.8571, abs=5e-4)

    # Macro Metrics
    # Macro P = (0.5 + 1.0) / 2 = 0.75
    # Macro R = (1.0 + 1.0) / 2 = 1.0
    # Macro F1 = (0.6666 + 1.0) / 2 = 0.8333
    assert agg["macro_precision"] == 0.75
    assert agg["macro_recall"] == 1.0
    assert agg["macro_f1"] == pytest.approx(0.8333, abs=5e-4)
