"""Reading one value two ways.

Every comparison in this migration was first written to log only the
differences, which makes an empty log mean either that the two agree or that
the comparison never ran. The gate for switching over is "no differences", so
that distinction is the whole point.
"""

import logging

from gpustack.utils.comparison import ComparisonTally, tally


def test_agreement_is_counted_not_only_silent(caplog):
    t = ComparisonTally("Thing")

    with caplog.at_level(logging.INFO):
        assert t.compare(1, 1, "a") is True

    assert (t.agreed, t.differed) == (1, 0)
    assert "agreed=1" in caplog.text


def test_a_difference_names_both_readings(caplog):
    t = ComparisonTally("Thing")

    with caplog.at_level(logging.INFO):
        assert t.compare(1, 2, "a") is False

    assert (t.agreed, t.differed) == (0, 1)
    assert "was 1, reads 2" in caplog.text


def test_nothing_to_say_is_not_a_difference():
    """The new reading having no answer -- rows not compiled yet, a lookup
    that found nothing -- would otherwise keep the gate shut for a reason
    unrelated to whether the two agree."""
    t = ComparisonTally("Thing")

    assert t.compare(1, None, "a") is True
    assert (t.agreed, t.differed) == (0, 0)


def test_nothing_to_say_can_be_a_difference_where_it_means_one():
    """Where the new reading always produces a value, None is an answer."""
    t = ComparisonTally("Thing")

    assert t.compare(1, None, "a", skip_if_none=False) is False
    assert t.differed == 1


def test_reporting_widens_so_a_busy_run_stays_readable(caplog):
    t = ComparisonTally("Thing")

    with caplog.at_level(logging.INFO):
        for _ in range(8):
            t.compare(1, 1, "a")

    # 1, 2, 4, 8 -- four lines for eight comparisons, and it keeps halving.
    assert caplog.text.count("Thing: agreed=") == 4


def test_one_tally_per_comparison_not_per_call():
    """The callers are functions with nothing to hang a tally on, and one per
    call would count to one and report every time."""
    assert tally("Some comparison") is tally("Some comparison")
    assert tally("Some comparison") is not tally("Another comparison")
