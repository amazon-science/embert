local batch_size = 3;
local epochs = 100;
local num_workers = 0;
local max_instances_in_memory = 5;
local traj_segment_size = 7;

{
    "dataset_reader": {
        "type": "alfred_split_supervised",
        "splits_path": "grolp/tests/fixtures/overfit.json",
        "data_root_path": "storage/data/alfred/json_feat_2.1.0/",
        "vis_feats_path": "alfred_maskrcnn",
        "instance_cache_dir": "split_instance_cache",
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
        "pretrained_model_path": "storage/models/pretrained/oscar-base-no-labels.bin",
        "actor": {
            "input_dim": 768,
            "hidden_dims": [768, -1],
            "num_layers": 2,
            "activations": ["gelu", "linear"],
            "dropout": [0.0, 0.0]
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
             "dropout": 0.0,
             "mem_len": 100,
             "n_head": 8
        },
        "state_repr_method": "hidden",
        "compute_confusion_matrix": true,
        "use_start_instr_loss": true,
        "ignore_keys": ["metadata"]
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
          "lr": 5e-5,
          "eps": 1e-8
        },
        "num_epochs": epochs
    },
}