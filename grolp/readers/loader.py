import logging
import math
import random
from functools import partial
from typing import Iterator
from typing import List

import torch
from allennlp.common import Params
from allennlp.data import DataLoader, Batch, Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.samplers.batch_sampler import BatchSampler as AllenBatchSampler
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler, \
    SubsetRandomSampler, SequentialSampler, RandomSampler

logger = logging.getLogger(__name__)


def alfred_collate(instances: List[Instance]):
    """
    This is the default function used to turn a list of `Instance`s into a `TensorDict`
    batch.
    """
    batch = Batch(instances)
    tensors = batch.as_tensor_dict()

    # token_ids: torch.Tensor,
    # token_type_ids: torch.Tensor,
    # text_mask: torch.Tensor,
    tensors["token_ids"] = tensors["instructions"]["tokens"]["token_ids"]
    tensors["token_type_ids"] = tensors["instructions"]["tokens"]["type_ids"]
    tensors["text_mask"] = tensors["instructions"]["tokens"]["mask"]

    del tensors["instructions"]

    return tensors


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.

    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.

    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    """

    def __init__(self, data, sort_key=None):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row) if sort_key is not None else row) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


def identity(x):
    return x


@AllenBatchSampler.register("custom_bucket")
class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular libraries like
        ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together examples with a similar
        size length to reduce the padding required for each batch while maintaining some noise
        through bucketing.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key=None,
                 bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        _bucket_size = batch_size * bucket_size_multiplier
        if hasattr(sampler, "__len__"):
            _bucket_size = min(_bucket_size, len(sampler))
        self.bucket_sampler = BatchSampler(sampler, _bucket_size, False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


def sort_by_num_actions(idx, dataset):
    return len(dataset.data[idx]["plan"]["low_actions"])


@DataLoader.register("alfred")
class AlfredDataLoader(DataLoader, TorchDataLoader):
    def __iter__(self):
        return TorchDataLoader.__iter__(self)

    def iter_instances(self) -> Iterator[Instance]:
        pass

    def index_with(self, vocab: Vocabulary) -> None:
        pass

    def set_target_device(self, device: torch.device) -> None:
        pass

    def __init__(self, dataset: Dataset, collate_fn=alfred_collate, **kwargs):
        kwargs["collate_fn"] = collate_fn

        if "batch_sampler" in kwargs and isinstance(kwargs["batch_sampler"], Params):
            # Assumptions: when batch sampler is specified assumes BucketSampler
            # shuffle True in bucket sampler
            shuffle = kwargs["batch_sampler"].get("shuffle", False)
            if shuffle:
                seq_sampler = RandomSampler(dataset)
            else:
                seq_sampler = SequentialSampler(dataset)

            kwargs["batch_sampler"] = BucketBatchSampler(
                seq_sampler,
                batch_size=kwargs["batch_sampler"].pop("batch_size"),
                drop_last=kwargs["batch_sampler"].pop("drop_last"),
                sort_key=partial(sort_by_num_actions, dataset=dataset)
            )
        TorchDataLoader.__init__(self, dataset, **kwargs)

    def __len__(self):
        return TorchDataLoader.__len__(self)
