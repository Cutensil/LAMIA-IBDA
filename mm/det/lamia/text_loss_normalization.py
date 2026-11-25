"""Call-local, asynchronous count normalization for text auxiliary losses."""

import torch


class TextLossNormalizer:
    """Batch counts in one collective and defer their use until loss return.

    Register every active scale/stage, including zero-count ranks. Once the
    last count is known, communication starts asynchronously; independent
    forward/loss computation continues without reading the result. Weighted
    loss numerators remain attached to autograd. Only counts are detached.
    This object belongs to one loss call, never to model/optimizer state.
    """

    def __init__(self, expected_counts):
        self.expected_counts = expected_counts
        self._keys = {}
        self._counts = []
        self._terms = []
        self._buffer = None
        self._work = None
        self._world_size = 1

    def register(self, key, count):
        if key in self._keys or len(self._counts) >= self.expected_counts:
            raise ValueError('Unexpected or duplicate text-loss count.')
        self._keys[key] = len(self._counts)
        self._counts.append(count.detach().float().reshape(()))
        if len(self._counts) == self.expected_counts:
            self._buffer = torch.stack(self._counts)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self._world_size = torch.distributed.get_world_size()
                self._work = torch.distributed.all_reduce(self._buffer, async_op=True)

    def add(self, name, numerator, key):
        if key not in self._keys:
            raise ValueError('Register a count before adding its loss numerator.')
        self._terms.append((name, numerator, self._keys[key]))

    def finalize(self, losses=None):
        """Wait only at the consumer, then apply exact global denominators."""
        if len(self._counts) != self.expected_counts:
            raise ValueError('Not all active text-loss counts were registered.')
        losses = {} if losses is None else losses
        if not self._counts:
            return losses
        if self._work is not None:
            self._work.wait()
        # Clamp globally BEFORE division by world size, including N < ranks.
        factors = (self._buffer.clamp_min(1.) / self._world_size).unbind()
        for name, numerator, index in self._terms:
            value = numerator / factors[index]
            losses[name] = losses.get(name, 0.) + value
        return losses
