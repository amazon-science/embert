local batch_size = 3;
local epochs = 50;
local num_workers = 0;
local max_instances_in_memory = 5;
local traj_segment_size = 7;
local num_objects_per_view = [36, 9, 9, 9];

{
    "dataset_reader": {
        "type": "embert_supervised_finegrained",
        "splits_path": "grolp/tests/fixtures/overfit.json",
        "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
        "vis_feats_path": "moca_maskrcnn",
        "instance_cache_dir": "embert_instance_cache",
        "mask_token_prob": 0,
        "mask_object_prob": 0,
        "num_objects_per_view": num_objects_per_view,
        "write_cache": true
    },
    "train_data_path": "train",
    "validation_data_path": "train",
    "vocabulary": {
        "type": "from_files",
        "directory": "storage/models/embert/vocab.tar.gz"
    },
    "model": {
        "type": "embert_alfred",
        # Recurrent VLN-BERT checkpoint have a dictionary structure. Make sure to extract "vln_bert" from them.
        # We stored a different checkpoint file
        "pretrained_model_path": "bert-base-uncased",
        "actor": {
            "input_dim": 768,
            "hidden_dims": [768, -1],
            "num_layers": 2,
            "activations": ["gelu", "linear"],
            "dropout": [0.0, 0.0]
        },
        "start_instr_predictor": {
            "input_dim": 768,
            "hidden_dims": [768, 1],
            "num_layers": 2,
            "activations": ["gelu", "linear"],
            "dropout": [0.0, 0.0]
        },
        "object_scorer": {
//            "type": "additive",
//            "vector_dim": 768,
//            "matrix_dim": 768,
//            "normalize": false
            "input_dim": 768,
            "hidden_dims": [768, 1],
            "num_layers": 2,
            "activations": ["gelu", "linear"],
            "dropout": [0.0, 0.0]
        },
        "state_encoder": {
             "n_layer":2,
             "d_model":768,
             "d_embed":100,
             "d_inner":512,
             "dropout": 0.0,
             "mem_len": 100,
             "n_head": 8,
             "use_object_embeddings": false
        },
        "num_objects_per_view": num_objects_per_view,
        "state_repr_method": "dot_product",
        "compute_confusion_matrix": true,
        "use_start_instr_loss": true,
        "ignore_keys": ["metadata"],
        "use_nav_receptacle_loss": false,
        "use_lm_loss": false,
        "use_vm_loss": false,
        "use_itm_loss": false
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
        "traj_segment_size": traj_segment_size,
        "optimizer": {
          "type": "huggingface_adamw",
          "weight_decay": 0.05,
          "parameter_groups": [
            [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
          ],
          "lr": 2e-5,
          "eps": 1e-8
        },
        "num_epochs": epochs,
        "validation_metric": "+overall"
    }
}