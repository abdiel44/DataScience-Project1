from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Sequence

import numpy as np

try:
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover - runtime guard
    class Sampler:  # type: ignore[no-redef]
        pass


class RecordingBatchSampler(Sampler[List[int]]):
    """Keep batches local to a recording while still shuffling batch order."""

    def __init__(
        self,
        recording_ids: Sequence[str],
        *,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self._groups: Dict[str, List[int]] = defaultdict(list)
        for idx, recording_id in enumerate(recording_ids):
            self._groups[str(recording_id)].append(int(idx))
        self._length = 0
        for indices in self._groups.values():
            full, rem = divmod(len(indices), self.batch_size)
            self._length += full
            if rem and not self.drop_last:
                self._length += 1

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        recording_ids = list(self._groups.keys())
        if self.shuffle:
            rng.shuffle(recording_ids)
        batches: List[List[int]] = []
        for recording_id in recording_ids:
            indices = list(self._groups[recording_id])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self._epoch += 1
        yield from batches
