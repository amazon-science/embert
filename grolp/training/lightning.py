import logging
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch.optim.lr_scheduler
from allennlp.common import Lazy
from allennlp.common import util as common_util
from allennlp.data import DataLoader
from allennlp.models.model import Model
from allennlp.training import Trainer as AllenNlpTrainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from pytorch_lightning import LightningDataModule, Callback as LightningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer import Trainer as LightningTrainer
from pytorch_lightning.utilities.cloud_io import get_filesystem
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class DataModule(LightningDataModule):
    def __init__(self, train, validation=None, test=None):
        super().__init__()

        self.train = train
        self.validation = validation
        self.test = test

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.validation

    def test_dataloader(self):
        return self.test


class LRSchedulerWrapper(_LRScheduler):
    def __init__(self, lr_scheduler: LearningRateScheduler):
        self._lr_scheduler = lr_scheduler

    @property
    def is_available(self):
        return self._lr_scheduler is not None

    @property
    def optimizer(self):
        return self._lr_scheduler.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._lr_scheduler.optimizer = optimizer

    def state_dict(self):
        return self._lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._lr_scheduler.load_state_dict(state_dict)

    def step(self, epoch=None):
        self._lr_scheduler.step()

    def get_lr(self):
        group = next(iter(self.optimizer.param_groups))
        return group["lr"]


@AllenNlpTrainer.register("lightning", constructor="from_partial_objects")
class PytorchLightningTrainer(AllenNlpTrainer):
    default_implementation = "lightning"

    def __init__(
            self,
            model: Model,
            optimizer: torch.optim.Optimizer,
            data_loader: DataLoader,
            distributed_params: Dict[str, int],
            recover: Optional[bool] = False,
            dry_run: Optional[bool] = False,
            patience: Optional[int] = None,
            validation_metric: Union[str, List[str]] = "loss",
            validation_data_loader: DataLoader = None,
            num_epochs: int = 20,
            serialization_dir: Optional[str] = None,
            grad_clipping: Optional[float] = None,
            learning_rate_scheduler: Optional[LearningRateScheduler] = None,
            momentum_scheduler: Optional[MomentumScheduler] = None,
            moving_average: Optional[MovingAverage] = None,
            callbacks: List[LightningCallback] = None,
            max_steps: int = None,
            num_gradient_accumulation_steps: int = 1,
            traj_segment_size=None,
            ignore_keys=("instructions", "metadata"),
            **kwargs
    ) -> None:
        super().__init__(serialization_dir)

        self.model = model
        self.model.optimizer = optimizer
        self.model.lr_scheduler = LRSchedulerWrapper(learning_rate_scheduler)
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader

        self.optimizer = optimizer
        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average
        self._callbacks = callbacks
        self.patience = patience
        self.validation_metric = validation_metric
        self.traj_segment_size = traj_segment_size
        self.ignore_keys = ignore_keys

        if recover:
            recover_checkpoint_path = self._max_ckpt_in_folder(serialization_dir)
        else:
            recover_checkpoint_path = None

        self.trainer = LightningTrainer(
            gradient_clip_val=grad_clipping,
            truncated_bptt_steps=traj_segment_size,
            accumulate_grad_batches=num_gradient_accumulation_steps,
            max_epochs=num_epochs,
            default_root_dir=serialization_dir,
            callbacks=self._callbacks,
            fast_dev_run=dry_run,
            resume_from_checkpoint=recover_checkpoint_path,
            max_steps=max_steps,
            **distributed_params,
            **kwargs
        )

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """

        training_results = self.trainer.fit(self.model, self.data_loader, self.validation_data_loader)
        final_metrics = self.trainer.callback_metrics
        if isinstance(training_results, int):
            final_metrics["training_state"] = "success" if training_results == 1 else "failure"
        if self.trainer.checkpoint_callback.best_model_score is not None:
            final_metrics["best_model_path"] = self.trainer.checkpoint_callback.best_model_path
            final_metrics[
                f"best_{self.trainer.checkpoint_callback.monitor}"] = self.trainer.checkpoint_callback.best_model_score.item()
        return final_metrics

    def test(self, data_loader):
        return self.trainer.test(self.model, data_loader)

    @contextmanager
    def get_checkpoint_state(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        raise NotImplementedError

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load("/path/to/model/weights.th"))`

        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        # Returns

        epoch: `int`
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        raise NotImplementedError

    @classmethod
    def from_partial_objects(
            cls,
            model: Model,
            serialization_dir: str,
            data_loader: DataLoader,
            recover: bool = False,
            dry_run: bool = False,
            validation_data_loader: DataLoader = None,
            patience: int = None,
            validation_metric: Union[str, List[str]] = None,
            num_epochs: int = 20,
            grad_clipping: float = None,
            max_steps: int = None,
            num_gradient_accumulation_steps: int = 1,
            no_grad: List[str] = None,
            optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
            learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
            momentum_scheduler: Lazy[MomentumScheduler] = None,
            moving_average: Lazy[MovingAverage] = None,
            num_serialized_models: int = 5,
            distributed_params: Dict[str, int] = None,
            traj_segment_size: int = None,
            ignore_keys: List[str] = ("instructions", "metadata"),
            **kwargs
    ) -> "Trainer":
        """
        This method exists so that we can have a documented method to construct this class using
        `FromParams`. If you are not using `FromParams` or config files, you can safely ignore this
        method.

        The reason we can't just use `__init__` with `FromParams` here is because there are
        sequential dependencies to this class's arguments.  Anything that has a `Lazy[]` type
        annotation needs something from one of the non-`Lazy` arguments.  The `Optimizer` needs to
        have the parameters from the `Model` before it's constructed, and the `Schedulers` need to
        have the `Optimizer`. Because of this, the typical way we construct things `FromParams`
        doesn't work, so we use `Lazy` to allow for constructing the objects sequentially.

        If you're not using `FromParams`, you can just construct these arguments in the right order
        yourself in your code and call the constructor directly.
        """

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)

        common_util.log_frozen_and_tunable_parameter_names(model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = max_steps

        moving_average_ = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        learning_rate_scheduler_ = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
            )
        )
        momentum_scheduler_ = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=optimizer_)
        )

        callbacks = []

        if validation_metric is not None:
            if validation_metric.startswith("+"):
                mode = "max"
                validation_metric = validation_metric.replace("+", "")
            else:
                mode = "min"
                validation_metric = validation_metric.replace("-", "")

            validation_metric = f"val_{validation_metric}"

            if patience is not None:
                callbacks.append(EarlyStopping(
                    monitor=validation_metric,
                    min_delta=0.00,
                    patience=patience,
                    verbose=False,
                    mode=mode
                ))
        else:
            # safe default
            validation_metric = "loss"
            mode = "min"

        callbacks.append(ModelCheckpoint(
            dirpath=serialization_dir,
            filename="model-{epoch}",
            save_top_k=num_serialized_models,
            verbose=True,
            monitor=validation_metric,
            mode=mode
        ))

        if distributed_params is not None:

            _distributed_params = {}

            for k, v in distributed_params.items():
                if v is not None:
                    _distributed_params[k] = v

            # we recreate the distributed params here after sanitising the user config params
            distributed_params = _distributed_params
        else:
            distributed_params = dict(
            )

        return cls(
            model,
            optimizer_,
            data_loader,
            max_steps=max_steps,
            dry_run=dry_run,
            recover=recover,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            moving_average=moving_average_,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            callbacks=callbacks,
            distributed_params=distributed_params,
            traj_segment_size=traj_segment_size,
            ignore_keys=ignore_keys,
            **kwargs
        )

    def _max_ckpt_in_folder(self, dir_path: Union[str, Path], name_key: str = 'model-epoch=') -> Optional[str]:
        """List up files in `dir_path` with `name_key`, then yield maximum suffix number.
        Args:
            dir_path: path of directory which may contain files whose name include `name_key`
            name_key: file name prefix
        Returns:
            None if no-corresponding-file else maximum suffix number
        """

        # check directory existence
        fs = get_filesystem(dir_path)
        if not fs.exists(dir_path):
            return None

        # check corresponding file existence
        files = [os.path.basename(f["name"]) for f in fs.listdir(dir_path)]
        files = [x for x in files if name_key in x]
        if len(files) == 0:
            return None

        max_checkpoint = (-1, None)

        for filename in files:
            name = filename.split(name_key)[-1]
            name = re.sub('[^0-9]', '', name)
            ckpt = int(name)

            if ckpt > max_checkpoint[0]:
                max_checkpoint = (ckpt, filename)

        return os.path.join(dir_path, max_checkpoint[1]) if max_checkpoint[1] is not None else None