import logging
import os
import shutil
from pathlib import Path
from typing import Union, Dict, Any

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.models import Archive
from allennlp.models.archival import _load_model, _load_dataset_readers, CONFIG_NAME, get_weights_path, \
    extracted_archive

logger = logging.getLogger(__name__)


def load_archive(
        archive_file: Union[str, Path],
        cuda_device: int = -1,
        overrides: Union[str, Dict[str, Any]] = "",
        weights_file: str = None,
) -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.

    # Parameters

    archive_file : `Union[str, Path]`
        The archive file to load the model from.
    cuda_device : `int`, optional (default = `-1`)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
        JSON overrides to apply to the unarchived `Params` object.
    weights_file : `str`, optional (default = `None`)
        The weights file to use.  If unspecified, weights.th in the archive_file will be used.
    """
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    tempdir = None
    try:
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            with extracted_archive(resolved_archive_file, cleanup=False) as tempdir:
                serialization_dir = tempdir

        if weights_file:
            weights_path = weights_file
        else:
            weights_path = get_weights_path(serialization_dir)

        # Load config
        config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)

        # Instantiate model and dataset readers. Use a duplicate of the config, as it will get consumed.
        dataset_reader, validation_dataset_reader = _load_dataset_readers(
            config.duplicate(), serialization_dir
        )
        model = _load_model(config.duplicate(), weights_path, serialization_dir, cuda_device)
    finally:
        if tempdir is not None:
            logger.info(f"removing temporary unarchived model dir at {tempdir}")
            shutil.rmtree(tempdir, ignore_errors=True)

    return Archive(
        model=model,
        config=config,
        dataset_reader=dataset_reader,
        validation_dataset_reader=validation_dataset_reader,
    )
