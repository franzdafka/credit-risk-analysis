import numpy as np

import credit_model as cm


def test_target_is_default_flag_with_30_percent_bad_rate() -> None:
    _, y = cm.load_data()
    assert set(y.unique()) == {0, 1}
    assert abs(y.mean() - 0.30) < 1e-9


def test_split_is_stratified_and_disjoint() -> None:
    X, y = cm.load_data()
    X_tr, X_te, y_tr, y_te = cm.split(X, y)
    assert len(X_te) == 200
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01


def test_expected_cost_trivial_policies() -> None:
    y = np.array([1, 0, 0, 0])
    pd_hat = np.array([0.5, 0.5, 0.5, 0.5])
    assert cm.expected_cost(y, pd_hat, 2.0) == cm.COST_BAD_APPROVED / 4  # approve all
    assert cm.expected_cost(y, pd_hat, -1.0) == 3 * cm.COST_GOOD_DECLINED / 4  # decline all


def test_bayes_threshold_for_5_to_1_costs() -> None:
    assert abs(cm.bayes_threshold() - 1 / 6) < 1e-12


def test_choose_threshold_recovers_perfect_separation() -> None:
    y = np.array([0] * 70 + [1] * 30)
    pd_hat = np.r_[np.full(70, 0.1), np.full(30, 0.9)]
    t = cm.choose_threshold(y, pd_hat)
    assert cm.expected_cost(y, pd_hat, t) == 0.0


def test_rating_grades_are_ordered() -> None:
    assert [cm.rating_grade(p) for p in (0.01, 0.07, 0.15, 0.3, 0.8)] == ["A", "B", "C", "D", "E"]


def test_saved_model_beats_trivial_policies_on_test() -> None:
    m = cm.load_artifact()["test_metrics"]
    assert m["auc_roc"] > 0.7
    assert m["brier"] < m["brier_base_rate"]
    assert m["cost_model"] < min(m["cost_approve_all"], m["cost_decline_all"])
