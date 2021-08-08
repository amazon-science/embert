from functools import partial
from typing import Iterable, Iterator

import numpy as np
import torch
from allennlp.common.util import shuffle_iterable
from allennlp.data import DatasetReader, Instance, Vocabulary, DataLoader, allennlp_collate
from allennlp.data.fields import ArrayField
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataset import IterableDataset, Dataset


def index_instance(instance, vocab, dataset_reader):
    dataset_reader.apply_token_indexers(instance)
    instance.index_fields(vocab)

    return instance


class PytorchIterableWrapper(IterableDataset):
    def __init__(self,
                 vocab: Vocabulary,
                 dataset_reader: DatasetReader,
                 data_info: str):
        self.dataset_reader = dataset_reader
        self.data_info = data_info
        self.vocab = vocab

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.dataset_reader._set_worker_info(worker_info)
        return map(
            partial(index_instance, vocab=self.vocab, dataset_reader=self.dataset_reader),
            self.dataset_reader.read(self.data_info)
        )


class PytorchWrapper(Dataset):
    def __init__(self,
                 vocab: Vocabulary,
                 dataset_reader: DatasetReader,
                 data_info: str):
        self.dataset_reader = dataset_reader
        self.data_info = data_info
        self.vocab = vocab

        self.data = list(self.dataset_reader.read(self.data_info))

    def __getitem__(self, item):
        datum = self.data[item]

        instance = self.dataset_reader.text_to_instance(datum["split"], datum)

        return index_instance(instance, vocab=self.vocab, dataset_reader=self.dataset_reader)

    def __len__(self):
        return len(self.data)


@DataLoader.register("lightning")
class LightningDataLoader(DataLoader, TorchDataLoader):
    def __iter__(self):
        return shuffle_iterable(TorchDataLoader.__iter__(self)) if self.shuffle else TorchDataLoader.__iter__(self)

    def iter_instances(self) -> Iterator[Instance]:
        pass

    def index_with(self, vocab: Vocabulary) -> None:
        pass

    def set_target_device(self, device: torch.device) -> None:
        pass

    def __init__(self, dataset: Dataset, collate_fn=allennlp_collate, **kwargs):
        kwargs["collate_fn"] = collate_fn
        shuffle = kwargs.get("shuffle", False)
        kwargs["shuffle"] = False
        TorchDataLoader.__init__(self, dataset, **kwargs)
        self.shuffle = shuffle

    def __len__(self):
        return TorchDataLoader.__len__(self)


class FakeReader(DatasetReader):
    def __init__(self, task, start=1, end=512):
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True)
        self.task = task
        self.start = start
        self.end = end

    def _read(self, file_path) -> Iterable[Instance]:
        data = list(self.shard_iterable(range(self.start, self.end)))

        for x in data:
            yield self.text_to_instance(x)

    def text_to_instance(self, x) -> Instance:
        x_field = ArrayField(np.array(x))

        return Instance({
            "x": x_field
        })
