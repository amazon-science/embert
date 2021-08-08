from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import DatasetReader


class TestAlfredReader(AllenNlpTestCase):
    def test_alfred_supervised_reader(self):
        reader = DatasetReader.from_params(Params({
            "type": "alfred_supervised",
            "splits_path": "grolp/tests/fixtures/tiny_splits.json",
            "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
            "vis_feats_path": "alfred_maskrcnn",
            "write_cache": True
        }))

        split_id = "train"
        instances = reader.read(split_id)

        i = 0
        for ex in instances:
            x = reader.text_to_instance(split_id, ex)
            i += 1

            if i == 5:
                break

        print(f"Total instances: {i}")

    def test_alfred_split_supervised_reader(self):
        reader = DatasetReader.from_params(Params({
            "type": "alfred_split_supervised",
            "splits_path": "grolp/tests/fixtures/tiny_splits.json",
            "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
            "vis_feats_path": "alfred_maskrcnn",
            "write_cache": True,
            "instance_cache_dir": "split_instance_cache"
        }))

        split_id = "train"
        instances = reader.read(split_id)

        i = 0
        for ex in instances:
            x = reader.text_to_instance(split_id, ex)
            i += 1

            if i == 5:
                break

        print(f"Total instances: {i}")

    def test_alfred_next_split_supervised_reader(self):
        reader = DatasetReader.from_params(Params({
            "type": "alfred_split_next_supervised",
            "splits_path": "grolp/tests/fixtures/tiny_splits.json",
            "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
            "vis_feats_path": "alfred_maskrcnn",
            "write_cache": True,
            "instance_cache_dir": "split_next_instance_cache"
        }))

        split_id = "train"
        instances = reader.read(split_id)

        i = 0
        for ex in instances:
            x = reader.text_to_instance(split_id, ex)
            i += 1

            if i == 5:
                break

        print(f"Total instances: {i}")

    def test_alfred_pretraining_reader(self):
        reader = DatasetReader.from_params(Params({
            "type": "alfred_pretraining",
            "splits_path": "grolp/tests/fixtures/tiny_splits.json",
            "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
            "vis_feats_path": "alfred_maskrcnn",
            "write_cache": True,
            "instance_cache_dir": "pretraining_instance_cache"
        }))

        split_id = "train"
        instances = reader.read(split_id)

        i = 0
        for ex in instances:
            x = reader.text_to_instance(split_id, ex)
            i += 1

            print(x)
            if i == 5:
                break

        print(f"Total instances: {i}")

    def test_alfred_finegrained_reader(self):
        reader = DatasetReader.from_params(Params({
            "type": "alfred_split_supervised_finegrained",
            "splits_path": "grolp/tests/fixtures/tiny_splits.json",
            "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
            "vis_feats_path": "alfred_maskrcnn",
            "write_cache": True,
            "instance_cache_dir": "split_finegrained_instance_cache"
        }))

        split_id = "train"
        instances = reader.read(split_id)

        i = 0
        for ex in instances:
            x = reader.text_to_instance(split_id, ex)
            i += 1

            print(x)
            if i == 5:
                break

        print(f"Total instances: {i}")

    def test_embert_reader(self):
        reader = DatasetReader.from_params(Params({
            "type": "embert_supervised_finegrained",
            "splits_path": "grolp/tests/fixtures/tiny_splits.json",
            "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
            "vis_feats_path": "alfred_maskrcnn",
            "write_cache": True,
            "instance_cache_dir": "embert_instance_cache",
            "mask_token_prob": 0.15
        }))

        split_id = "train"
        instances = reader.read(split_id)

        i = 0
        for ex in instances:
            x = reader.text_to_instance(split_id, ex)
            i += 1

            print(x)
            if i == 5:
                break

        print(f"Total instances: {i}")
