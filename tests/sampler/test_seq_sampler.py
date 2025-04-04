import pytest
import numpy as np


from genlm.control.potential import Potential
from genlm.control.sampler.sequence import Importance, SMC, Sequences, SequenceModel
from genlm.control.sampler.token import DirectTokenSampler

from hypothesis import strategies as st, settings, given
from conftest import (
    weighted_set,
    weighted_sequence,
    double_weighted_sequence,
    WeightedSet,
)


@pytest.fixture
def default_unit_sampler():
    sequences = ["a", "b", "c"]
    weights = [1, 2, 3]
    p = WeightedSet(sequences, weights)
    return DirectTokenSampler(p)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence))
async def test_importance(S):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = Importance(unit_sampler, n_particles)

    sequences = await sampler.infer()
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(double_weighted_sequence))
async def test_importance_with_critic(S):
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 10
    sampler = Importance(unit_sampler, n_particles=n_particles, critic=critic)
    sequences = await sampler.infer()

    logeps = await p.prefix([])
    for seq, logw in sequences:
        logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(len(seq))])
        assert np.isclose(logw, logZ + logeps + await critic.score(seq))


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence), st.floats(min_value=0, max_value=1))
async def test_smc(S, ess_threshold):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = SMC(unit_sampler, n_particles, ess_threshold)

    sequences = await sampler.infer()
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(st.floats(min_value=0, max_value=1))
async def test_smc_with_critic(ess_threshold):
    seqs = ["0", "00", "1"]
    weights1 = [3.0, 2.0, 1.0]
    weights2 = [1.0, 2.0, 3.0]

    p = WeightedSet(seqs, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(seqs, weights2)

    n_particles = 500
    sampler = SMC(unit_sampler, n_particles, ess_threshold=ess_threshold, critic=critic)

    sequences = await sampler.infer()

    intersection_ws = [w1 * w2 for w1, w2 in zip(weights1, weights2)]
    assert len(sequences) == n_particles
    assert np.isclose(
        np.exp(sequences.log_ml), sum(intersection_ws), atol=0.5, rtol=0.05
    )


@st.composite
def smc_params(draw, item_sampler, max_seq_len=5, max_size=5):
    S = draw(weighted_set(item_sampler, max_seq_len, max_size))
    stop_point = draw(st.integers(min_value=1, max_value=max_seq_len))
    return S, stop_point


@pytest.mark.asyncio
@settings(deadline=None)
@given(smc_params(double_weighted_sequence))
async def test_smc_weights(params):
    S, stop_point = params
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 10
    sampler = SMC(
        unit_sampler,
        critic=critic,
        n_particles=n_particles,
        ess_threshold=0,  # don't resample since that would reset weights
        max_tokens=stop_point,
    )

    sequences = await sampler.infer()

    logeps = await p.prefix([])
    for seq, logw in sequences:
        logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(len(seq))])
        twist = await critic.score(seq)
        assert np.isclose(logw, logZ + logeps + twist)


def test_sequences():
    sequences = Sequences(
        contexts=[["a", "b", "c"], ["a", "b", "d"]],
        log_weights=[np.log(1), np.log(9)],
        log_probs=[np.log(1), np.log(9)],
    )

    assert np.allclose(sequences.normalized_weights, [0.1, 0.9])

    assert sequences[0] == (["a", "b", "c"], np.log(1))
    assert sequences[1] == (["a", "b", "d"], np.log(9))

    assert len(sequences) == 2
    assert sequences.logp == np.log(1) + np.log(9)
    assert sequences.log_total == np.log(10)

    sequences._repr_html_()
    sequences.show()


async def test_sequence_model_invalid_start_weight():
    class MockPotential(Potential):
        async def prefix(self, context):
            if not context:
                return -np.inf
            return 0

        async def complete(self, context):
            return 0

    unit_sampler = DirectTokenSampler(MockPotential([0]))
    with pytest.raises(ValueError, match="Start weight.*"):
        await SequenceModel(unit_sampler).start()


def test_sequence_model_str_for_serialization(default_unit_sampler):
    SequenceModel(default_unit_sampler).string_for_serialization()


def test_sequence_sampler_max_tokens(default_unit_sampler):
    sampler = Importance(default_unit_sampler, n_particles=10)
    sampler.max_tokens = 5
    assert sampler.max_tokens == 5


def test_sequence_sampler_invalid_args(default_unit_sampler):
    with pytest.raises(ValueError, match="n_particles must be greater than 0"):
        Importance(default_unit_sampler, n_particles=0)

    with pytest.raises(ValueError, match="n_particles must be greater than 0"):
        SMC(default_unit_sampler, n_particles=0, ess_threshold=0.5)

    with pytest.raises(ValueError, match="ess_threshold must be between 0 and 1.0"):
        SMC(default_unit_sampler, n_particles=10, ess_threshold=1.5)


def test_sequence_sampler_sample_not_implemented(default_unit_sampler):
    sampler = Importance(default_unit_sampler, n_particles=10)
    with pytest.raises(NotImplementedError):
        sampler.sample()

    sampler = SMC(default_unit_sampler, n_particles=10, ess_threshold=0.5)
    with pytest.raises(NotImplementedError):
        sampler.sample()
