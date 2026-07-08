import numpy as np
from torch.utils.data import Sampler
from circa.data.dataset import SpatialNeighborhoodDataset

class InterleavedSlideBatchSampler(Sampler[list[int]]):
    """
    Create batches within each slide, then shuffle the list of batches so slides are interleaved.

    Important: indices yielded are DATASET indices (0..len(dataset)-1).
    """

    def __init__(
        self,
        slide_to_dsidx: list[np.ndarray],
        batch_size: int,
        *,
        shuffle_within: bool = True,
        shuffle_batches: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.slide_to_dsidx = [np.asarray(x, dtype=np.int64) for x in slide_to_dsidx]
        self.batch_size = int(batch_size)
        self.shuffle_within = shuffle_within
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        if drop_last:
            self._len = sum(len(x) // self.batch_size for x in self.slide_to_dsidx)
        else:
            self._len = sum((len(x) + self.batch_size - 1) // self.batch_size for x in self.slide_to_dsidx)

    def __iter__(self):
        all_batches: list[list[int]] = []

        for idxs0 in self.slide_to_dsidx:
            if idxs0.size == 0:
                continue

            idxs = idxs0.copy()
            if self.shuffle_within:
                self.rng.shuffle(idxs)

            n_full = idxs.size // self.batch_size
            end = n_full * self.batch_size

            for b in range(n_full):
                all_batches.append(idxs[b * self.batch_size : (b + 1) * self.batch_size].tolist())

            if not self.drop_last and end < idxs.size:
                all_batches.append(idxs[end:].tolist())

        if self.shuffle_batches:
            self.rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return self._len


class SlideBlockBatchSampler(Sampler[list[int]]):
    """
    Create fixed-size blocks within each slide, then combine randomly selected
    slide blocks into full batches.

    Important: indices yielded are DATASET indices (0..len(dataset)-1).

    Example:
        batch_size=256, block_size=16
        -> each full batch contains 16 slide blocks of 16 samples each.
    """

    def __init__(
        self,
        slide_to_dsidx: list[np.ndarray],
        batch_size: int,
        block_size: int,
        *,
        shuffle_within: bool = True,
        shuffle_blocks: bool = True,
        shuffle_batches: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.slide_to_dsidx = [np.asarray(x, dtype=np.int64) for x in slide_to_dsidx]
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.batch_size % self.block_size != 0:
            raise ValueError("batch_size must be divisible by block_size.")

        self.blocks_per_batch = self.batch_size // self.block_size
        self.shuffle_within = shuffle_within
        self.shuffle_blocks = shuffle_blocks
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        # Number of blocks produced per slide.
        if drop_last:
            n_blocks = sum(len(x) // self.block_size for x in self.slide_to_dsidx)
        else:
            n_blocks = sum(
                (len(x) + self.block_size - 1) // self.block_size
                for x in self.slide_to_dsidx
                if len(x) > 0
            )

        if drop_last:
            self._len = n_blocks // self.blocks_per_batch
        else:
            self._len = (n_blocks + self.blocks_per_batch - 1) // self.blocks_per_batch

    def __iter__(self):
        all_blocks: list[list[int]] = []

        # First create blocks within each slide.
        for idxs0 in self.slide_to_dsidx:
            if idxs0.size == 0:
                continue

            idxs = idxs0.copy()
            if self.shuffle_within:
                self.rng.shuffle(idxs)

            n_full = idxs.size // self.block_size
            end = n_full * self.block_size

            for block_i in range(n_full):
                start = block_i * self.block_size
                stop = start + self.block_size
                all_blocks.append(idxs[start:stop].tolist())

            if not self.drop_last and end < idxs.size:
                all_blocks.append(idxs[end:].tolist())

        # Then randomly mix slide blocks before constructing full batches.
        if self.shuffle_blocks:
            self.rng.shuffle(all_blocks)

        all_batches: list[list[int]] = []

        n_full_batches = len(all_blocks) // self.blocks_per_batch
        end_block = n_full_batches * self.blocks_per_batch

        for batch_i in range(n_full_batches):
            start = batch_i * self.blocks_per_batch
            stop = start + self.blocks_per_batch

            batch: list[int] = []
            for block in all_blocks[start:stop]:
                batch.extend(block)

            all_batches.append(batch)

        if not self.drop_last and end_block < len(all_blocks):
            batch = []
            for block in all_blocks[end_block:]:
                batch.extend(block)
            all_batches.append(batch)

        if self.shuffle_batches:
            self.rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return self._len


class EqualExposureSlideBatchSampler(Sampler[list[int]]):
    """
    Sample the same number of batches from each slide per epoch.

    Important: indices yielded are DATASET indices (0..len(dataset)-1).

    This differs from InterleavedSlideBatchSampler:
      - It does NOT exhaust all cells from all slides.
      - Each non-empty slide contributes exactly `batches_per_slide` batches per epoch.
      - Cells are sampled within slide, optionally with replacement.
      - This makes the training objective closer to averaging over slides rather than cells.

    Recommended for highly imbalanced slides, e.g. 1k vs 19k cells.
    """

    def __init__(
        self,
        slide_to_dsidx: list[np.ndarray],
        batch_size: int,
        *,
        batches_per_slide: int | None = None,
        steps_per_epoch: int | None = None,
        replacement: bool = True,
        shuffle_batches: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        self.slide_to_dsidx = [np.asarray(x, dtype=np.int64) for x in slide_to_dsidx]
        self.batch_size = int(batch_size)
        self.replacement = bool(replacement)
        self.shuffle_batches = bool(shuffle_batches)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

        self.nonempty_slides = [s for s, idxs in enumerate(self.slide_to_dsidx) if idxs.size > 0]
        self.n_nonempty_slides = len(self.nonempty_slides)

        if self.n_nonempty_slides == 0:
            raise ValueError("No non-empty slides found in slide_to_dsidx.")

        if batches_per_slide is not None and steps_per_epoch is not None:
            raise ValueError("Specify only one of `batches_per_slide` or `steps_per_epoch`, not both.")

        if batches_per_slide is None and steps_per_epoch is None:
            # Default: one pass over the median-sized slide.
            # This is a reasonable default, but I usually recommend setting this explicitly.
            nonempty_sizes = np.array(
                [self.slide_to_dsidx[s].size for s in self.nonempty_slides],
                dtype=np.int64,
            )
            median_n = int(np.median(nonempty_sizes))
            if drop_last:
                batches_per_slide = max(1, median_n // self.batch_size)
            else:
                batches_per_slide = max(1, (median_n + self.batch_size - 1) // self.batch_size)

        if steps_per_epoch is not None:
            steps_per_epoch = int(steps_per_epoch)
            if steps_per_epoch <= 0:
                raise ValueError("`steps_per_epoch` must be positive.")

            # Make the epoch as evenly slide-balanced as possible.
            self.steps_per_epoch = steps_per_epoch
            base = steps_per_epoch // self.n_nonempty_slides
            rem = steps_per_epoch % self.n_nonempty_slides

            self._slide_counts = {s: base for s in self.nonempty_slides}
            if rem > 0:
                extra_slides = self.rng.choice(self.nonempty_slides, size=rem, replace=False)
                for s in extra_slides:
                    self._slide_counts[s] += 1

            self.batches_per_slide = None

        else:
            batches_per_slide = int(batches_per_slide)
            if batches_per_slide <= 0:
                raise ValueError("`batches_per_slide` must be positive.")

            self.batches_per_slide = batches_per_slide
            self.steps_per_epoch = self.n_nonempty_slides * self.batches_per_slide
            self._slide_counts = {
                s: self.batches_per_slide for s in self.nonempty_slides
            }

        self._len = self.steps_per_epoch

    def __iter__(self):
        # Build one epoch's slide schedule with equal, or nearly equal, exposure.
        slide_schedule: list[int] = []
        for s in self.nonempty_slides:
            slide_schedule.extend([s] * self._slide_counts[s])

        if self.shuffle_batches:
            self.rng.shuffle(slide_schedule)

        for s in slide_schedule:
            idxs = self.slide_to_dsidx[s]
            n = idxs.size

            if self.replacement:
                # Best for equal slide exposure, especially when some slides are small.
                batch = self.rng.choice(
                    idxs,
                    size=self.batch_size,
                    replace=True,
                )
                yield batch.tolist()

            else:
                # Without replacement within a sampled batch.
                # If a slide has fewer than batch_size cells:
                #   - drop_last=True: skip this batch
                #   - drop_last=False: return a smaller batch
                if n >= self.batch_size:
                    batch = self.rng.choice(
                        idxs,
                        size=self.batch_size,
                        replace=False,
                    )
                    yield batch.tolist()
                else:
                    if self.drop_last:
                        continue
                    else:
                        yield idxs.copy().tolist()

    def __len__(self):
        return self._len
        

def build_slide_to_dsidx(dataset, slide_key: str = "slide_id") -> list[np.ndarray]:
    """
    Returns a list where element s contains dataset indices (0..len(dataset)-1)
    whose center cell belongs to slide code s.

    Works correctly for train/val splits because it uses dataset.indices.
    """
    if not hasattr(dataset, "indices"):
        raise AttributeError("dataset must have `indices` (dataset idx -> adata idx).")

    if not hasattr(dataset, "_obs") or slide_key not in dataset._obs:
        raise KeyError(f"dataset._obs[{slide_key!r}] not found. Make sure you included it in obs_keys.")

    # slide codes for all adata rows
    slide_codes_all = dataset._obs[slide_key]  # length N_obs, int64 codes
    adata_idx_for_ds = dataset.indices         # length len(dataset), values in [0, N_obs)

    slide_codes_ds = slide_codes_all[adata_idx_for_ds]  # length len(dataset)

    n_slides = int(slide_codes_all.max()) + 1  # assumes codes are -1..S-1; strings->category should avoid -1
    slide_to_dsidx = []
    for s in range(n_slides):
        slide_to_dsidx.append(np.flatnonzero(slide_codes_ds == s).astype(np.int64, copy=False))
    return slide_to_dsidx