import os
from argparse import ArgumentParser

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from vision import utils
from vision.coco_eval import CocoEvaluator
from vision.coco_utils import get_coco_api_from_dataset
from vision.engine import _get_iou_types
from vision.readers import AI2ThorDataset


class MaskRCNN(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        # load an instance segmentation model pre-trained on COCO
        if "inference_params" in kwargs:
            inference_params = kwargs.get("inference_params")
            self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                               **inference_params)
        else:
            self.detector = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.hparams.num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.detector.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = self.hparams.hidden_size
        # and replace the mask predictor with a new one
        self.detector.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                   hidden_layer,
                                                                   self.hparams.num_classes)
        # These are initialised at the beginning of the training process
        self.valid_evaluator = None
        self.test_evaluator = None

    def configure_optimizers(self):
        # construct an optimizer
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.hparams.lr,
                                    momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=self.hparams.step_size,
                                                       gamma=self.hparams.gamma)

        return [optimizer], [lr_scheduler]

    def forward(self, image, target):
        loss_dict = self.detector(image, target)

        return loss_dict

    def on_validation_start(self) -> None:
        if self.valid_evaluator is None:
            coco = get_coco_api_from_dataset(self.val_dataloader().dataset, "validation")
            iou_types = _get_iou_types(self.detector)
            self.valid_evaluator = CocoEvaluator(coco, iou_types)
        self.valid_evaluator.reset()

    def on_test_start(self) -> None:
        if self.test_evaluator is None:
            iou_types = _get_iou_types(self.detector)
            coco = get_coco_api_from_dataset(self.test_dataloader().dataset, "test")
            self.test_evaluator = CocoEvaluator(coco, iou_types)
        self.test_evaluator.reset()

    def on_validation_end(self) -> None:
        self.valid_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        self.valid_evaluator.accumulate()
        self.valid_evaluator.summarize()

    def on_test_end(self) -> None:
        self.test_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        self.test_evaluator.accumulate()
        self.test_evaluator.summarize()

    def training_step(self, batch, batch_idx):
        image, target = batch
        loss_dict = self.forward(image, target)

        losses = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict, on_epoch=True)

        return losses

    def inference_step(self, batch, split_key):
        image, target = batch
        outputs = self.detector(image)

        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(target, outputs)}
        if split_key == "val":
            self.valid_evaluator.update(res)
        else:
            self.test_evaluator.update(res)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, "test")


def main(args):
    dataset = AI2ThorDataset(args.train_file, args.vocab_file)
    num_classes = len(dataset.vocab)
    # This is to make sure that we can change the MaskRCNN backbone and ROI predictor
    args.num_classes = num_classes
    dataset_valid = AI2ThorDataset(args.valid_file, args.vocab_file)
    dataset_test = AI2ThorDataset(args.test_file, args.vocab_file)

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    # batch_size should be equal to 1 when evaluating
    data_loader_valid = DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    model = MaskRCNN(**vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename='maskrcnn-{epoch:02d}',
        save_top_k=-1
    )

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=args.save_dir, callbacks=[checkpoint_callback])

    trainer.fit(model, train_dataloader=data_loader, val_dataloaders=data_loader_valid)

    trainer.test(model, data_loader_test)


if __name__ == "__main__":
    parser = ArgumentParser()

    # SGD parameters
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # LR scheduler
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)

    # Model
    parser.add_argument("--hidden_size", type=int, default=256)

    # Datasets
    parser.add_argument("--train_file", type=str, default="storage/data/alfred/images/train/metadata.json")
    parser.add_argument("--valid_file", type=str, default="storage/data/alfred/images/valid_seen/metadata.json")
    parser.add_argument("--test_file", type=str, default="storage/data/alfred/images/valid_unseen/metadata.json")

    # Training
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--vocab_file", type=str, default="configs/ai2thor_vocab.json")
    parser.add_argument("--save_dir", type=str, default="storage/models/vision/maskrcnn")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
