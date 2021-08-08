local gpu_batch_size = 128;
local num_gpus = 1;
local effective_batch_size = gpu_batch_size*num_gpus;
local epochs = 30;
local num_workers = 4;

{
    "dataset_reader": {
        "type": "alfred_pretraining",
        "splits_path": "storage/data/alfred/splits/oct21.json",
        "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
        "vis_feats_path": "alfred_maskrcnn",
        "instance_cache_dir": "pretraining_instance_cache",
        "write_cache": false
    },
    "train_data_path": "train",
    "validation_data_path": "valid_seen",
    "vocabulary": {
        "type": "from_files",
        "directory": "storage/models/embert/vocab.tar.gz"
    },
    "model": {
        "type": "embert_pretrain",
        # Recurrent VLN-BERT checkpoint have a dictionary structure. Make sure to extract "vln_bert" from them.
        # We stored a different checkpoint file
        "pretrained_model_path": "bert-base-uncased"
    },
    "data_loader": {
        "type": "alfred",
        "batch_sampler": {
            "batch_size": gpu_batch_size,
            "drop_last": false,
            "shuffle": true
        },
        "num_workers": num_workers
    },
    "validation_data_loader": {
        "type": "alfred",
        "batch_sampler": {
            "batch_size": gpu_batch_size,
            "drop_last": false,
            "shuffle": false
        },
        "num_workers": num_workers
    },
    "distributed": {
        "gpus": std.range(0, num_gpus - 1),
        "precision": 16,
        "amp_level": "01"
    },
    "trainer": {
        "type": "lightning",
        "optimizer": {
          "type": "huggingface_adamw",
          "weight_decay": 0.05,
          "parameter_groups": [
               [["embert.*bias", "embert.*LayerNorm\\.weight", "embert.*layer_norm\\.weight"], {"weight_decay": 0}],
          ],
          "lr": 2e-5
        },
        "learning_rate_scheduler": {
            "type": "linear_with_warmup",
            "warmup_steps": 0
        },
        "num_serialized_models": 5, // saves all models
        "grad_clipping": 1.0,
        "num_epochs": epochs,
        "validation_metric": "-loss",
        "num_gradient_accumulation_steps": 4
    },
}