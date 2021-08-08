import multiprocessing
from argparse import ArgumentParser
from functools import partial

from allennlp.common import Params
from tqdm import tqdm

from grolp.readers.alfred import *


def process_instance(ex, dataset_reader, split=None):
    if split is not None:
        return dataset_reader.text_to_instance(split, ex)

    return dataset_reader.text_to_instance(ex)


def main(args):
    for split in ("valid_seen", "valid_unseen", "train"):
        params = Params.from_file(args.config)
        params = params.pop("dataset_reader")
        # forces the dataset reader to ignore the cache and always recreate the instances
        params["write_cache"] = True
        dataset_reader = DatasetReader.from_params(params)
        examples = [x for x in tqdm(dataset_reader.read(split), desc=f"Reading examples for {split}...")]

        with multiprocessing.Pool(args.num_workers) as pool:
            iterator = pool.imap_unordered(
                partial(process_instance, dataset_reader=dataset_reader, split=split),
                examples
            )
            for x in tqdm(iterator, desc=f"Processing examples for split {split}", total=len(examples)):
                pass


def process_with_reader(dataset_reader, num_workers, split):
    dataset_reader.write_cache = True
    examples = [x for x in tqdm(dataset_reader.read(split), desc=f"Reading examples for {split}...")]

    with multiprocessing.Pool(num_workers) as pool:
        iterator = pool.imap_unordered(
            partial(process_instance, dataset_reader=dataset_reader, split=None),
            examples
        )
        for x in tqdm(iterator, desc=f"Processing examples for split {split}", total=len(examples)):
            pass
    dataset_reader.write_cache = False


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--num_workers", default=0, type=int)

    args = parser.parse_args()

    main(args)
