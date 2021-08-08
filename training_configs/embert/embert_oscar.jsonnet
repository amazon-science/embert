local gpu_batch_size = 32;
local eval_gpu_batch_size = 8;
local num_gpus = 1;
local effective_batch_size = gpu_batch_size*num_gpus;
local epochs = 30;
local num_workers = 3;
local total_training_instances = 21011;
local traj_segment_size = 12;
local max_traj_length = 150;
{
    "dataset_reader": {
        "type": "alfred_supervised",
        "splits_path": "storage/data/alfred/splits/oct21.json",
        "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
        "vis_feats_path": "alfred_maskrcnn",
        "max_traj_length": max_traj_length,
        "instance_cache_dir": "instance_cache"
    },
    "train_data_path": "train",
    "validation_data_path": "valid_seen",
    "vocabulary": {
        "type": "from_files",
        "directory": "storage/models/embert/vocab.tar.gz"
    },
    "model": {
        "type": "embert_alfred",
        # Recurrent VLN-BERT checkpoint have a dictionary structure. Make sure to extract "vln_bert" from them.
        # We stored a different checkpoint file
        "pretrained_model_path": "storage/models/pretrained/oscar-base-no-labels.bin",
        "actor": {
            "input_dim": 768,
            "hidden_dims": [768, -1],
            "num_layers": 2,
            "activations": ["gelu", "linear"],
            "dropout": [0.1, 0.0]
        },
        "object_scorer": {
            "type": "additive",
            "vector_dim": 768,
            "matrix_dim": 768,
            "normalize": false // makes sure that we do not apply softmax over the attention weights
        },
        "state_encoder": {
             "n_layer":2,
             "d_model":768,
             "d_embed":100,
             "d_inner":512,
             "dropout": 0.1,
             "mem_len": 100,
             "n_head": 8
        },
        "compute_confusion_matrix": true,
        "state_repr_method": "dot_product"
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
            "batch_size": eval_gpu_batch_size,
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
        "num_serialized_models": -1, // saves all models
        "grad_clipping": 1.0,
        "traj_segment_size": traj_segment_size,
        "num_epochs": epochs,
        "validation_metric": "-loss"
    },
}