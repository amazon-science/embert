import base64
import json
import re

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


class AI2ThorDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, vocab_file="configs/ai2thor_vocab.json"):
        # this is the path to the metadata for the current split (called "metadata.json")
        self.filepath = filepath

        with open(self.filepath) as in_file:
            self.examples = json.load(in_file)

        # We don't need to add image normalisation because it is handled internally by MaskRCNN
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        with open(vocab_file) as in_file:
            self.vocab = json.load(in_file)

    def _index_class(self, class_):
        if "Sliced" not in class_ and "Cracked" not in class_:
            return self.vocab[class_]

        parts = re.findall('[A-Z][a-z]*', class_)
        obj_class = "".join(parts[:-1])

        return self.vocab.get(obj_class)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, image_id):
        annotations_file = self.examples[image_id]

        with open(annotations_file) as in_file:
            example = json.load(in_file)

        # load images ad masks
        img_path = example["file_name"]
        img = Image.open(img_path).convert("RGB")

        annotations = example["annotations"]

        boxes = []
        masks = []
        classes = []

        for obj in annotations:
            # degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            # if degenerate_boxes.any():
            if obj["bbox"][2] <= obj["bbox"][0] or obj["bbox"][3] <= obj["bbox"][1]:
                continue

            boxes.append(obj["bbox"])
            mask = np.frombuffer(base64.b64decode(obj["mask"].encode("utf-8")), dtype=np.uint8).reshape(
                *obj["mask_shape"])
            masks.append(mask)
            classes.append(self._index_class(obj["category"]))

        if not classes:
            # raise ValueError(f"No bounding boxes available for this image: {img_path}")
            return None

        num_objects = len(classes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(classes, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["image_path"] = img_path
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = self.transforms(img)

        return img, target


if __name__ == "__main__":
    reader = AI2ThorDataset("storage/data/alfred/images/train/metadata.json")

    for i, ex in enumerate(reader):
        if i == 5:
            break

        print(i)
