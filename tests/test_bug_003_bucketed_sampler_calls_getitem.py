import pytest


@pytest.mark.cpu
def test_bucketed_sampler_should_not_call_getitem_for_sizing():
    """
    BucketedBatchSampler now uses metadata to determine grid sizes without calling __getitem__.
    
    This was fixed by adding _get_max_grid_size_from_task_metadata() which extracts size 
    information directly from dataset.tasks metadata, avoiding the need to call __getitem__.
    """
    from sci_arc.data.dataset import BucketedBatchSampler

    class ExplodingDataset:
        """Dataset whose __getitem__ must not be called during bucketing."""

        def __init__(self, n: int = 10):
            self._n = n
            # Provide tasks metadata (what sampler *should* use for sizing)
            self.tasks = [
                {
                    "task_id": str(i),
                    "train": [
                        {"input": [[0]], "output": [[0]]},
                    ],
                    "test": [
                        {"input": [[0]], "output": [[0]]},
                    ],
                }
                for i in range(n)
            ]

        def __len__(self):
            return self._n

        def __getitem__(self, idx: int):
            raise RuntimeError(
                "BucketedBatchSampler should not call __getitem__ just to compute a size bucket."
            )

    ds = ExplodingDataset(n=25)

    # If BucketedBatchSampler uses metadata only, this should construct successfully.
    _ = BucketedBatchSampler(dataset=ds, batch_size=4, bucket_boundaries=[2, 5, 10], drop_last=False)
