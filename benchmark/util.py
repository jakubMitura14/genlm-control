import time
import os
import pickle
from genlm_control.util import fast_sample_lazyweights
from genlm_control.sampler.set import SetSampler
from genlm_control.sampler.token import TokenSampler


def normalize_and_sample_token(logws, draw=None):
    logps = logws.normalize()
    if draw is None:
        # fast sampling from logps using gumbel-max trick
        token = fast_sample_lazyweights(logps)
    else:
        token = draw(logps.exp().materialize())
    return token, logps[token]


class CachedDirectTokenSampler(TokenSampler):
    def __init__(self, potential, cache_file=None, save_every=100):
        super().__init__(target=potential)
        self.potential = potential
        self.cache = {}
        self.cache_file = cache_file
        self.save_every = save_every

        # Load cache from file if it exists
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"Loaded cache with {len(self.cache)} entries from {cache_file}")
            except Exception as e:
                print(f"Error loading cache from {cache_file}: {e}")

    def _save_cache(self):
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)

    async def sample(self, context, draw=None):
        cache_key = tuple(context)
        if cache_key in self.cache:
            logws = self.cache[cache_key]
        else:
            logws = await self.potential.logw_next(context)
            self.cache[cache_key] = logws
            if len(self.cache) % self.save_every == 0:
                self._save_cache()
        token, logp = normalize_and_sample_token(logws, draw)
        return token, logws.sum(), logp

    async def cleanup(self):
        self._save_cache()

    def __del__(self):
        self._save_cache()


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
