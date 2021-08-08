"""
The `train` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.
"""

import argparse
import logging
import os
from datetime import datetime
from os import PathLike
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import torch
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import TrainModel
from allennlp.common import Params, Lazy, Registrable
from allennlp.common import logging as common_logging
from allennlp.common import util as common_util
from allennlp.common.plugins import import_plugins
from allennlp.data import DataLoader as AllenNlpDataLoader
from allennlp.data import DatasetReader, Vocabulary
from allennlp.models.archival import CONFIG_NAME, verify_include_in_archive, archive_model
from allennlp.models.model import Model
from allennlp.training import util as training_util
from allennlp.training.trainer import Trainer
from overrides import overrides

from grolp.readers.lightning import PytorchWrapper
from grolp.training.lightning import DataModule, PytorchLightningTrainer
from scripts.process_dataset import main

logger = logging.getLogger(__name__)


@Subcommand.register("ltrain")
class LightningTrain(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Train a model.")

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "--preprocess",
            action="store_true",
            help="Preprocesses the dataset",
        )

        subparser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="Preprocesses the dataset",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """
    if args.preprocess:
        logger.info("Preprocessing enabled. Starting instance cache generation")
        start_time = datetime.now()
        main(SimpleNamespace(**dict(config=args.param_path, num_workers=args.num_workers)))
        end_time = datetime.now() - start_time
        logger.info(f"Processing finished after: {end_time}")

    logger.info("Starting training...")

    train_model_from_file(
        parameter_filename=args.param_path,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
        recover=args.recover,
        force=args.force,
        include_package=args.include_package,
        dry_run=args.dry_run
    )


def train_model_from_file(
        parameter_filename: Union[str, PathLike],
        serialization_dir: Union[str, PathLike],
        overrides: Union[str, Dict[str, Any]] = "",
        recover: bool = False,
        force: bool = False,
        include_package: List[str] = None,
        dry_run: bool = False
) -> Optional[Model]:
    """
    A wrapper around [`train_model`](#train_model) which loads the params from a file.

    # Parameters

    parameter_filename : `str`
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : `str`
        The directory in which to save results and logs. We just pass this along to
        [`train_model`](#train_model).
    overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
        A JSON string or a dict that we will use to override values in the input parameter file.
    recover : `bool`, optional (default=`False`)
        If `True`, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see `Model.from_archive`.
    force : `bool`, optional (default=`False`)
        If `True`, we will overwrite the serialization directory if it already exists.
    node_rank : `int`, optional
        Rank of the current node in distributed training
    include_package : `str`, optional
        In distributed mode, extra packages mentioned will be imported in training workers.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.

    # Returns

    best_model : `Optional[Model]`
        The model with the best epoch weights or `None` if in dry run.
    """
    # Load the experiment config from a file and pass it to `train_model`.
    params = Params.from_file(parameter_filename, overrides)

    return _train_model(
        params=params,
        serialization_dir=serialization_dir,
        recover=recover,
        force=force,
        include_package=include_package,
        dry_run=dry_run
    )


def _train_model(
        params: Params,
        serialization_dir: Union[str, PathLike],
        recover: bool = False,
        force: bool = False,
        include_package: List[str] = None,
        dry_run: bool = False,
        file_friendly_logging: bool = False,
) -> Optional[Model]:
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    training_util.create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    include_in_archive = params.pop("include_in_archive", None)
    verify_include_in_archive(include_in_archive)

    common_util.prepare_environment(params)

    import_plugins()
    include_package = include_package or None
    for package_name in include_package:
        common_util.import_module_and_submodules(package_name)
    distributed_params = params.pop("distributed", None)

    train_loop = LightningTrainModel.from_params(
        params=params,
        serialization_dir=serialization_dir,
        recover=recover,
        dry_run=dry_run,
        distributed_params=distributed_params
    )

    final_metrics = train_loop.run()

    train_loop.finish(final_metrics)


class LightningTrainModel(Registrable):
    """
    This class exists so that we can easily read a configuration file with the `allennlp train`
    command.  The basic logic is that we call `train_loop =
    TrainModel.from_params(params_from_config_file)`, then `train_loop.run()`.  This class performs
    very little logic, pushing most of it to the `Trainer` that has a `train()` method.  The
    point here is to construct all of the dependencies for the `Trainer` in a way that we can do
    it using `from_params()`, while having all of those dependencies transparently documented and
    not hidden in calls to `params.pop()`.  If you are writing your own training loop, you almost
    certainly should not use this class, but you might look at the code for this class to see what
    we do, to make writing your training loop easier.

    In particular, if you are tempted to call the `__init__` method of this class, you are probably
    doing something unnecessary.  Literally all we do after `__init__` is call `training.train()`.  You
    can do that yourself, if you've constructed a `Trainer` already.  What this class gives you is a
    way to construct the `Trainer` by means of a config file.  The actual constructor that we use
    with `from_params` in this class is `from_partial_objects`.  See that method for a description
    of all of the allowed top-level keys in a configuration file used with `allennlp train`.
    """

    default_implementation = "lightning"

    def __init__(self, serialization_dir: str, model: Model, trainer: PytorchLightningTrainer,
                 data_module: DataModule, evaluate_on_test: bool) -> None:
        super().__init__()
        self.serialization_dir = serialization_dir
        self.model = model
        self.trainer = trainer
        self.data_module = data_module
        self.evaluate_on_test = evaluate_on_test

    def run(self) -> Dict[str, Any]:
        return self.trainer.train()

    def finish(self, metrics: Dict[str, Any]):
        if self.data_module.test is not None and self.evaluate_on_test:
            logger.info("The model will be evaluated using the best epoch weights.")
            test_metrics = self.trainer.test(self.data_module.test)[0]

            for key, value in test_metrics.items():
                metrics[key] = float(value)
        elif self.data_module.test is not None:
            logger.info(
                "To evaluate on the test set after training, pass the "
                "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
            )

        for k, v in metrics.items():
            metrics[k] = float(v) if not isinstance(v, str) else v

        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )
        best_ckpt = torch.load(metrics['best_model_path'], map_location="cpu")

        best_path = os.path.join(self.serialization_dir, "best.th")
        logging.info(f"Saving best model weights in path: {best_path}")
        torch.save(best_ckpt, best_path)

        archive_model(self.serialization_dir, "best.th")

    @classmethod
    def from_partial_objects(
            cls,
            serialization_dir: str,
            dry_run: bool,
            recover: bool,
            distributed_params: Dict[str, int],
            dataset_reader: DatasetReader,
            train_data_path: Any,
            model: Lazy[Model],
            data_loader: Lazy[AllenNlpDataLoader],
            trainer: Lazy[Trainer],
            vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary),
            validation_dataset_reader: DatasetReader = None,
            validation_data_path: Any = None,
            validation_data_loader: Lazy[AllenNlpDataLoader] = None,
            test_data_path: Any = None,
            evaluate_on_test: bool = False,
            **kwargs
    ) -> "TrainModel":
        """
        This method is intended for use with our `FromParams` logic, to construct a `TrainModel`
        object from a config file passed to the `allennlp train` command.  The arguments to this
        method are the allowed top-level keys in a configuration file (except for the first three,
        which are obtained separately).

        You *could* use this outside of our `FromParams` logic if you really want to, but there
        might be easier ways to accomplish your goal than instantiating `Lazy` objects.  If you are
        writing your own training loop, we recommend that you look at the implementation of this
        method for inspiration and possibly some utility functions you can call, but you very likely
        should not use this method directly.

        The `Lazy` type annotations here are a mechanism for building dependencies to an object
        sequentially - the `TrainModel` object needs data, a model, and a training, but the model
        needs to see the data before it's constructed (to create a vocabulary) and the training needs
        the data and the model before it's constructed.  Objects that have sequential dependencies
        like this are labeled as `Lazy` in their type annotations, and we pass the missing
        dependencies when we call their `construct()` method, which you can see in the code below.

        # Parameters

        serialization_dir: `str`
            The directory where logs and model archives will be saved.

            In a typical AllenNLP configuration file, this parameter does not get an entry as a
            top-level key, it gets passed in separately.

        dataset_reader: `DatasetReader`
            The `DatasetReader` that will be used for training and (by default) for validation.

        train_data_path: `str`
            The file (or directory) that will be passed to `dataset_reader.read()` to construct the
            training data.

        model: `Lazy[Model]`
            The model that we will train.  This is lazy because it depends on the `Vocabulary`;
            after constructing the vocabulary we call `model.construct(vocab=vocabulary)`.

        data_loader: `Lazy[DataLoader]`
            The data_loader we use to batch instances from the dataset reader at training and (by
            default) validation time. This is lazy because it takes a dataset in it's constructor.

        training: `Lazy[Trainer]`
            The `Trainer` that actually implements the training loop.  This is a lazy object because
            it depends on the model that's going to be trained.

        vocabulary: `Lazy[Vocabulary]`, optional (default=`Lazy(Vocabulary)`)
            The `Vocabulary` that we will use to convert strings in the data to integer ids (and
            possibly set sizes of embedding matrices in the `Model`).  By default we construct the
            vocabulary from the instances that we read.

        datasets_for_vocab_creation: `List[str]`, optional (default=`None`)
            If you pass in more than one dataset but don't want to use all of them to construct a
            vocabulary, you can pass in this key to limit it.  Valid entries in the list are
            "train", "validation" and "test".

        validation_dataset_reader: `DatasetReader`, optional (default=`None`)
            If given, we will use this dataset reader for the validation data instead of
            `dataset_reader`.

        validation_data_path: `str`, optional (default=`None`)
            If given, we will use this data for computing validation metrics and early stopping.

        validation_data_loader: `Lazy[DataLoader]`, optional (default=`None`)
            If given, the data_loader we use to batch instances from the dataset reader at
            validation and test time. This is lazy because it takes a dataset in it's constructor.

        test_data_path: `str`, optional (default=`None`)
            If given, we will use this as test data.  This makes it available for vocab creation by
            default, but nothing else.

        evaluate_on_test: `bool`, optional (default=`False`)
            If given, we will evaluate the final model on this data at the end of training.  Note
            that we do not recommend using this for actual test data in every-day experimentation;
            you should only very rarely evaluate your model on actual test data.

        batch_weight_key: `str`, optional (default=`""`)
            The name of metric used to weight the loss on a per-batch basis.  This is only used
            during evaluation on final test data, if you've specified `evaluate_on_test=True`.
        """
        # Train data loader.
        vocabulary_ = vocabulary.construct()
        dataset_reader_ = PytorchWrapper(vocabulary_, dataset_reader, train_data_path)
        data_loaders = {
            "train": data_loader.construct(dataset=dataset_reader_)
        }

        # Validation data loader.
        if validation_data_path is not None:
            validation_dataset_reader = validation_dataset_reader or dataset_reader
            validation_dataset_reader_ = PytorchWrapper(vocabulary_, validation_dataset_reader,
                                                        validation_data_path)
            if validation_data_loader is None:
                validation_data_loader = data_loader

            validation_data_loader_ = validation_data_loader.construct(dataset=validation_dataset_reader_)
            data_loaders["validation"] = validation_data_loader_

        # Test data loader.
        if test_data_path is not None:
            test_dataset_reader = validation_dataset_reader or dataset_reader
            test_dataset_reader_ = PytorchWrapper(vocabulary_, test_dataset_reader, test_data_path)
            data_loaders["test"] = validation_data_loader.construct(dataset=test_dataset_reader_)

        model_ = model.construct(vocab=vocabulary_, serialization_dir=serialization_dir)

        vocabulary_path = os.path.join(serialization_dir, "vocabulary")
        vocabulary_.save_to_files(vocabulary_path)

        data_module = DataModule(
            **data_loaders
        )
        trainer_ = trainer.construct(
            model=model_,
            data_loader=data_loaders["train"],
            validation_data_loader=data_loaders.get("validation"),
            recover=recover,
            dry_run=dry_run,
            distributed_params=distributed_params,
            **kwargs
        )
        assert trainer_ is not None

        return cls(
            serialization_dir=serialization_dir,
            model=model_,
            trainer=trainer_,
            data_module=data_module,
            evaluate_on_test=evaluate_on_test
        )


LightningTrainModel.register("lightning", constructor="from_partial_objects")(LightningTrainModel)
