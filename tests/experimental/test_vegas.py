import pytest
import asyncio
import numpy as np
from conftest import MockPotential
from hypothesis import given, strategies as st, settings, reject

from genlm_control.experimental.vegas import (
    RejectionSampler,
    GumbelMaxRejectionSampler,
    AdaptiveRejectionSampler,
    GumbelMaxAdaptiveRejectionSampler,
    ClippedAdaptiveRejectionSampler,
)

RUN_MC_TESTS = False


@st.composite
def V_size(draw):
    # Generate a vocabulary of size <=4.
    return draw(st.integers(min_value=1, max_value=4))


@st.composite
def cont_weights(draw, V_size, min_p=1e-3):
    # Generate a list of floats for each token in the vocabulary (and EOS).
    ws = draw(st.lists(st.floats(min_p, 1), min_size=V_size + 1, max_size=V_size + 1))
    Z = sum(ws)
    ps = [w / Z for w in ws]
    return ps


@st.composite
def bool_weights(draw, V_size):
    # Generate a list of booleans for each token in the vocabulary (and EOS).
    bws = draw(st.lists(st.booleans(), min_size=V_size + 1, max_size=V_size + 1))
    if not any(bws):
        # Need at least one valid token.
        reject()
    return bws


@st.composite
def params(draw, min_p=1e-3):
    vocab_size = draw(V_size())
    b_weights = draw(bool_weights(vocab_size))
    c_weights = draw(cont_weights(vocab_size, min_p))
    return [bytes([i]) for i in range(vocab_size)], b_weights, c_weights


async def assert_monte_carlo_close(
    sampler_cls, params, N, equality_opts={}, sampler_opts={}
):
    vocab, b_weights, c_weights = params
    potential = MockPotential(vocab, np.log(c_weights))
    condition = MockPotential(vocab, np.log(b_weights))

    sampler = sampler_cls(potential, condition, **sampler_opts)

    want = await sampler.target.logw_next([])
    have = await sampler._monte_carlo([], N)

    assert np.isclose(np.exp(want.sum()), np.exp(have.sum()), **equality_opts)


async def assert_variance_reduction(sampler_cls, params, N1, N2, K, sampler_opts={}):
    # Check that the variance of the logZ estimate is reduced when using
    # a larger number of samples.
    assert N1 < N2

    vocab, b_weights, c_weights = params
    potential = MockPotential(vocab, np.log(c_weights))
    condition = MockPotential(vocab, np.log(b_weights))

    sampler = sampler_cls(potential, condition, **sampler_opts)

    N1s = await asyncio.gather(*[sampler._monte_carlo([], N1) for _ in range(K)])
    Zs_N1 = np.array([np.exp(have.sum()) for have in N1s])
    N2s = await asyncio.gather(*[sampler._monte_carlo([], N2) for _ in range(K)])
    Zs_N2 = np.array([np.exp(have.sum()) for have in N2s])

    var_N1 = np.var(Zs_N1)
    var_N2 = np.var(Zs_N2)

    # If both variances are extremely small (close to machine epsilon),
    # the test should pass regardless of their relative values
    epsilon = 1e-30
    if var_N1 < epsilon and var_N2 < epsilon:
        return

    assert var_N1 > var_N2


@pytest.mark.asyncio
@pytest.mark.skipif(not RUN_MC_TESTS, reason="Skipping Monte Carlo tests")
@settings(deadline=None, max_examples=25)
@given(params(min_p=0.1), st.integers(min_value=1, max_value=2))
async def test_rejection_sampler(params, M):
    await assert_monte_carlo_close(
        sampler_cls=RejectionSampler,
        params=params,
        N=10000,
        equality_opts={"rtol": 1e-2, "atol": 1e-2},
        sampler_opts={"M": M},
    )

    await assert_variance_reduction(
        sampler_cls=RejectionSampler,
        params=params,
        N1=100,
        N2=1000,
        K=20,
        sampler_opts={"M": M},
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not RUN_MC_TESTS, reason="Skipping Monte Carlo tests")
@settings(deadline=None, max_examples=25)
@given(params(min_p=0.1))
async def test_gumbelmax_rejection_sampler(params):
    await assert_monte_carlo_close(
        sampler_cls=GumbelMaxRejectionSampler,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
    )

    await assert_variance_reduction(
        sampler_cls=GumbelMaxRejectionSampler, params=params, N1=100, N2=1000, K=20
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not RUN_MC_TESTS, reason="Skipping Monte Carlo tests")
@settings(deadline=None, max_examples=25)
@given(params())
async def test_adaptive_rejection_sampler(params):
    await assert_monte_carlo_close(
        sampler_cls=AdaptiveRejectionSampler,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
    )

    await assert_variance_reduction(
        sampler_cls=AdaptiveRejectionSampler, params=params, N1=100, N2=1000, K=20
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not RUN_MC_TESTS, reason="Skipping Monte Carlo tests")
@settings(deadline=None)
@given(params())
async def test_gumbelmax_adaptive_rejection_sampler(params):
    await assert_monte_carlo_close(
        sampler_cls=GumbelMaxAdaptiveRejectionSampler,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
    )

    await assert_variance_reduction(
        sampler_cls=GumbelMaxAdaptiveRejectionSampler,
        params=params,
        N1=100,
        N2=1000,
        K=20,
    )


@pytest.mark.asyncio
# @pytest.mark.skipif(not RUN_MC_TESTS, reason="Skipping Monte Carlo tests")
@settings(deadline=None)
@given(
    params(),
    st.floats(min_value=0.4, max_value=1.0),
    st.floats(min_value=0.4, max_value=1.0),
)
async def test_clipped_adaptive_rejection_sampler(params, top_p1, top_p2):
    await assert_monte_carlo_close(
        sampler_cls=ClippedAdaptiveRejectionSampler,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
        sampler_opts={"top_p1": top_p1, "top_p2": top_p2},
    )

    await assert_variance_reduction(
        sampler_cls=ClippedAdaptiveRejectionSampler,
        params=params,
        N1=100,
        N2=1000,
        K=20,
        sampler_opts={"top_p1": top_p1, "top_p2": top_p2},
    )


sampler_clses = [
    GumbelMaxAdaptiveRejectionSampler,
    AdaptiveRejectionSampler,
    GumbelMaxRejectionSampler,
    RejectionSampler,
    ClippedAdaptiveRejectionSampler,
]


default_potential = MockPotential(
    [bytes([i]) for i in range(4)],
    np.log([0.1, 0.2, 0.2, 0.1, 0.4]),
)


default_condition = MockPotential(
    [bytes([i]) for i in range(4)],
    [0, 0, float("-inf"), float("-inf"), 0],
)


@pytest.mark.asyncio
@pytest.mark.parametrize("sampler_cls", sampler_clses)
async def test_verbosity(sampler_cls):
    sampler = sampler_cls(
        potential=default_potential,
        condition=default_condition,
    )
    await sampler.sample([], verbosity=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("sampler_cls", sampler_clses)
async def test_logging(sampler_cls):
    sampler = sampler_cls(
        potential=default_potential, condition=default_condition, log_stats=True
    )

    num_requests = 5
    contexts = [[] for _ in range(num_requests)]

    await asyncio.gather(*[sampler.sample(ctx) for ctx in contexts])

    stats = sampler.get_stats()

    assert len(stats["total_times"]) == num_requests
    assert len(stats["logws_times"]) == num_requests
    assert len(stats["calls"]) == num_requests
    assert len(stats["contexts"]) == num_requests

    sampler._reset_stats()
    empty_stats = sampler.get_stats()
    assert len(empty_stats["total_times"]) == 0
    assert len(empty_stats["calls"]) == 0
    assert len(empty_stats["contexts"]) == 0
    assert len(empty_stats["logws_times"]) == 0
