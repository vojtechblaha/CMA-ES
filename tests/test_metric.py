from pfn_cmaes.metrics import CounterfactualImprovementMetric


def test_metric_positive_for_improvement():
    metric = CounterfactualImprovementMetric()
    assert metric(10.0, 5.0) > 0.0


def test_metric_negative_for_regression():
    metric = CounterfactualImprovementMetric()
    assert metric(5.0, 10.0) < 0.0
