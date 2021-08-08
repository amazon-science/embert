local batch_size = 3;
local epochs = 20;
local num_workers = 0;

{
    "dataset_reader": {
        "type": "alfred_pretraining",
        "splits_path": "grolp/tests/fixtures/overfit.json",
        "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
        "vis_feats_path": "alfred_maskrcnn",
        "write_cache": true,
        "instance_cache_dir": "pretraining_instance_cache"
    },
    "train_data_path": "train",
    "validation_data_path": "train",
    "vocabulary": {
        "type": "from_files",
        "directory": "storage/models/embert/vocab.tar.gz"
    },
    "model": {
        "type": "embert_pretrain",
        # Recurrent VLN-BERT checkpoint have a dictionary structure. Make sure to extract "vln_bert" from them.
        # We stored a different checkpoint file
        "pretrained_model_path": "bert-base-uncased",
        "use_itm_loss": true
    },
    "data_loader": {
        "type": "alfred",
        "num_workers": num_workers,
        "batch_sampler": {
            "batch_size": batch_size,
            "drop_last": false,
            "shuffle": true
        }
    },
    "validation_data_loader": {
        "type": "alfred",
        "num_workers": num_workers,
        "batch_sampler": {
            "batch_size": batch_size,
            "drop_last": false,
            "shuffle": false
        }
    },
    "trainer": {
        "type": "lightning",
        "optimizer": {
          "type": "huggingface_adamw",
          "lr": 2e-5,
          "eps": 1e-8
        },
        "num_epochs": epochs
    },
}