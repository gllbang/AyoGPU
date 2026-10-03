import pytest

from scripts.evaluate_herasys_answers import normalize, token_f1


def test_answer_metric_preserves_counts_and_does_not_reward_verbose_answers():
    assert normalize("The Eiffel Tower!") == "eiffel tower"
    assert token_f1("Paris", "Paris") == 1
    assert token_f1("Paris Paris Lyon", "Paris Lyon") == pytest.approx(4 / 5)
    assert token_f1("Paris is in France", "Paris") == pytest.approx(2 / 5)
    assert token_f1("", "Paris") == 0
