#/bin/bash

PYTHONPATH=. python scripts/generate_maskrcnn_horizon0.py --batch_size 8 --num_workers 4 --cuda_device 0 --split_id valid_seen --features_folder moca_maskrcnn_36_18_18_18 --model_checkpoint storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt;
PYTHONPATH=. python scripts/generate_maskrcnn_horizon0.py --batch_size 8 --num_workers 4 --cuda_device 0 --split_id valid_unseen --features_folder moca_maskrcnn_36_18_18_18 --model_checkpoint storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt;
PYTHONPATH=. python scripts/generate_maskrcnn_horizon0.py --batch_size 8 --num_workers 4 --cuda_device 0 --split_id train --features_folder moca_maskrcnn_36_18_18_18 --model_checkpoint storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt;