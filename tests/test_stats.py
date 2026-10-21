from bank_assistant.stats import bootstrap_ci, paired_bootstrap


def test_bootstrap_interval_contains_the_mean():
    e = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    assert e.low <= e.value <= e.high and e.value == 0.7


def test_paired_bootstrap_detects_a_real_difference_and_not_noise():
    a = [0] * 50 + [1] * 50
    better = [1] * 30 + [0] * 20 + [1] * 50          # fixes 30 of the 50 failures
    assert paired_bootstrap(a, better).significant
    assert not paired_bootstrap(a, a).significant
