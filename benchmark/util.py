import time
from genlm_control.sampler.token import TokenSampler


class TimedTokenSampler(TokenSampler):
    """A wrapper around any TokenSampler that times sampling operations.

    This sampler implements the exact same interface as the wrapped sampler,
    but tracks and reports timing information for sample operations.

    Args:
        sampler (TokenSampler): The token sampler to wrap and time
    """

    def __init__(self, sampler):
        super().__init__(target=sampler.target)
        self.sampler = sampler
        self.sample_times = []

    async def sample(self, context, draw=None):
        """Sample a token while timing the operation.

        This method wraps the underlying sampler's sample method and records timing information.

        Args:
            context (list[int]): A sequence of tokens in the target potential's vocabulary.
            draw (callable, optional): A callable that draws a sample from a distribution.

        Returns:
            (token, weight, logp): Same return values as the wrapped sampler.
        """
        start_time = time.time()
        token, weight, logp = await self.sampler.sample(context, draw)
        elapsed = time.time() - start_time

        self.sample_times.append(elapsed)

        return token, weight, logp

    def get_timing_stats(self):
        """Get timing statistics as a dictionary.

        Returns:
            Dict containing timing statistics
        """
        if not self.sample_times:
            return {"sample_count": 0, "total_time": 0}

        return {
            "sample_count": len(self.sample_times),
            "total_time": sum(self.sample_times),
            "avg_time": sum(self.sample_times) / len(self.sample_times),
            "max_time": max(self.sample_times),
            "min_time": min(self.sample_times),
            "samples_per_sec": len(self.sample_times) / sum(self.sample_times),
            "times": self.sample_times,
        }

    async def start_weight(self):
        return await self.sampler.start_weight()

    async def cleanup(self):
        """Clean up the sampler."""
        await self.sampler.cleanup()
