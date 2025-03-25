import time
import asyncio
from genlm_control.util import fast_sample_lazyweights
from genlm_control.sampler.set import SetSampler
from genlm_control.sampler.token import TokenSampler


def make_sampler(sampler_name, llm, bool_cfg, sampler_args, time_sampler=False):
    if sampler_name == "eager":
        from genlm_control.sampler import EagerSetSampler
        from benchmark.util import LoggedSetTokenSampler

        return LoggedSetTokenSampler(
            EagerSetSampler(llm, bool_cfg, **sampler_args), log_stats=time_sampler
        )
    elif sampler_name == "swar":
        from genlm_control.experimental.vegas import GumbelMaxAdaptiveRejectionSampler

        return GumbelMaxAdaptiveRejectionSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )

    elif sampler_name == "clipped-swar":
        from genlm_control.experimental.vegas import ClippedAdaptiveRejectionSampler

        return ClippedAdaptiveRejectionSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )

    elif sampler_name == "swor":
        from genlm_control.experimental.vegas import WithoutReplacementSampler

        return WithoutReplacementSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "top-k":
        from genlm_control.sampler import TopKSetSampler
        from benchmark.util import LoggedSetTokenSampler

        return LoggedSetTokenSampler(
            TopKSetSampler(llm, bool_cfg, **sampler_args), log_stats=time_sampler
        )
    elif sampler_name == "rejection":
        from genlm_control.experimental.vegas import RejectionSampler

        return RejectionSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "lm":
        from genlm_control.sampler import DirectTokenSampler

        return DirectTokenSampler(llm)

    elif sampler_name == "direct":
        from genlm_control.sampler import DirectTokenSampler

        return DirectTokenSampler(llm * bool_cfg.coerce(llm, f=b"".join))

    elif sampler_name == "saved-direct":
        return SavedDirectTokenSampler(llm, bool_cfg.coerce(llm, f=b"".join))
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def normalize_and_sample_token(logws, draw=None):
    logps = logws.normalize()
    if draw is None:
        # fast sampling from logps using gumbel-max trick
        token = fast_sample_lazyweights(logps)
    else:
        token = draw(logps.exp().materialize())
    return token, logps[token]


class SavedDirectTokenSampler(TokenSampler):
    def __init__(self, potential, condition):
        self.potential = potential
        self.condition = condition
        self._reset_stats()
        super().__init__(target=potential * condition)

    async def sample(self, context, draw=None):
        W1, W2 = await asyncio.gather(
            self.potential.logw_next(context), self.condition.logw_next(context)
        )

        potential_ws = W1.weights[self.target.v1_idxs]
        condition_ws = W2.weights[self.target.v2_idxs]
        logws = self.target.make_lazy_weights(potential_ws + condition_ws)

        token, logp = normalize_and_sample_token(logws, draw)

        logZ = logws.sum()

        self.stats["results"].append((token, logZ))
        self.stats["weights"].append(
            (potential_ws, condition_ws, self.target.vocab_eos)
        )
        self.stats["contexts"].append(context)

        return token, logZ, logp

    def get_stats(self):
        return self.stats

    def _reset_stats(self):
        self.stats = {"results": [], "weights": [], "contexts": []}

    async def cleanup(self):
        pass


class LoggedSetTokenSampler(TokenSampler):
    def __init__(self, set_sampler, log_stats=False):
        assert isinstance(set_sampler, SetSampler)
        super().__init__(set_sampler.target)
        self.set_sampler = set_sampler
        self.log_stats = log_stats
        if log_stats:
            self._reset_stats()

    async def sample(self, context, draw=None):
        if self.log_stats:
            start_time = time.process_time()
            sample_id = self._get_sample_id()
            self._contexts[sample_id] = list(context)

        logws, logp_s = await self.set_sampler.sample_set(context, draw=draw)
        token, logp_t = normalize_and_sample_token(logws, draw)

        if self.log_stats:
            self._total_times[sample_id] = time.process_time() - start_time
            self._sets[sample_id] = logws.exp().materialize().trim()
            self._results[sample_id] = token

        return token, logws.sum(), logp_s + logp_t

    def _get_sample_id(self):
        self._sample_id += 1
        return self._sample_id

    def _reset_stats(self):
        self._sample_id = -1
        self._total_times = {}
        self._contexts = {}
        self._sets = {}
        self._results = {}

    def get_stats(self):
        if not self.log_stats:
            raise ValueError("Logging is not enabled")
        keys = ["total_times", "results", "sets", "contexts"]
        stats = {k: [] for k in keys}
        for sample_id in range(self._sample_id + 1):
            for key in keys:
                stats[key].append(getattr(self, f"_{key}")[sample_id])
        return stats

    async def cleanup(self):
        await self.set_sampler.cleanup()
