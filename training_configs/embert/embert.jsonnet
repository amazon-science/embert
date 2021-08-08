local gpu_batch_size = 8;
local eval_gpu_batch_size = 8;
local num_gpus = 1;
local effective_batch_size = gpu_batch_size*num_gpus;
local epochs = 20;
local num_workers = 4;
local total_training_instances = 21011;
local traj_segment_size = 16;
local max_traj_length = 150;
{
    "dataset_reader": {
        "type": "embert_supervised_finegrained",
        "splits_path": "storage/data/alfred/splits/oct21.json",
        "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
        "vis_feats_path": "moca_maskrcnn",
        "instance_cache_dir": "embert_instance_cache",
        "mask_token_prob": -0.0,
        "mask_object_prob": -0.0,
        "max_traj_length": max_traj_length
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
            "input_dim": 768,
            "hidden_dims": [768, 1],
            "num_layers": 2,
            "activations": ["gelu", "linear"],
            "dropout": [0.1, 0.0]
        },
        "state_encoder": {
             "n_layer":2,
             "d_model":768,
             "d_embed":100,
             "d_inner":512,
             "dropout": 0.1,
             "mem_len": 200,
             "n_head": 8
        },
        "compute_confusion_matrix": true,
        "state_repr_method": "dot_product",
        "use_nav_receptacle_loss": false,
        "use_start_instr_loss": true,
        "use_lm_loss": false,
        "use_vm_loss": false,
        "use_itm_loss": false
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
        "num_serialized_models": 5,
        "grad_clipping": 1.0,
        "traj_segment_size": traj_segment_size,
        "num_epochs": epochs,
        "validation_metric": "+overall"
    },
}