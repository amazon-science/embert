#/bin/bash

PYTHONPATH=. python scripts/extract_vision_dataset.py --num_workers 8 --split_id valid_seen --output_file storage/data/alfred/valid_seen.jsonl.gz;
PYTHONPATH=. python scripts/extract_vision_dataset.py --num_workers 8 --split_id valid_unseen --output_file storage/data/alfred/valid_unseen.jsonl.gz;
PYTHONPATH=. python scripts/extract_vision_dataset.py --num_workers 8 --split_id train --output_file storage/data/alfred/train.jsonl.gz;