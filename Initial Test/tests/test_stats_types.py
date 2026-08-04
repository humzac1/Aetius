from stats.types import CaseObservations, PairedCaseData, case_rate, paired_rate_diff


def test_case_observations_rate():
    obs = CaseObservations("c1", "fam", (1, 0, 1, 1))
    assert obs.n == 4
    assert obs.successes == 3
    assert obs.rate == 0.75


def test_case_rate_is_unweighted_mean_of_case_rates_not_pooled():
    # One case with 2 runs at rate 1.0, one case with 100 runs at rate 0.0.
    # Pooled: 2/102 ~= 0.02. Cluster-respecting (unweighted mean of case
    # rates): (1.0 + 0.0)/2 = 0.5. This is the whole point of case_rate.
    high = CaseObservations("high", "fam", (1, 1))
    low = CaseObservations("low", "fam", tuple([0] * 100))
    assert case_rate([high, low]) == 0.5


def test_paired_case_data_rejects_case_id_mismatch():
    a = CaseObservations("a", "fam", (1,))
    b = CaseObservations("b", "fam", (0,))
    try:
        PairedCaseData("a", "fam", a, b)
        assert False, "should have raised"
    except ValueError:
        pass


def test_paired_rate_diff():
    d1 = PairedCaseData("c1", "fam", CaseObservations("c1", "fam", (0, 0)), CaseObservations("c1", "fam", (1, 1)))
    d2 = PairedCaseData("c2", "fam", CaseObservations("c2", "fam", (1, 1)), CaseObservations("c2", "fam", (1, 0)))
    # c1 diff = 1.0 - 0.0 = 1.0; c2 diff = 0.5 - 1.0 = -0.5; mean = 0.25
    assert paired_rate_diff([d1, d2]) == 0.25
