# EmBERT: A Transformer Model for Embodied, Language-guided Visual Task Completion

We present Embodied BERT (EmBERT), a transformer-based model which can attend to high-dimensional, multi-modal inputs
across long temporal horizons for language-conditioned task completion. Additionally, we bridge the gap between
successful object-centric navigation models used for non-interactive agents and the language-guided visual task
completion benchmark, ALFRED, by introducing object navigation targets for EmBERT training. We achieve competitive
performance on the ALFRED benchmark, and EmBERT marks the first transformer-based model to successfully handle the
long-horizon, dense, multi-modal histories of ALFRED, and the first ALFRED model to utilize object-centric navigation
targets.

In this repository, we provide the entire codebase which is used for training and evaluating EmBERT performance on the
ALFRED dataset. It's mostly based on [AllenNLP](https://allennlp.org/)
and [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
therefore it's inherently easily to extend.

## Setup

We used Anaconda for our experiments. Please create an anaconda environment and then install the project dependencies
with the following command:

```bash
pip install -r requirements.txt
```

We currently use a custom version of PyTorch-Lightning that is bundled in the current codebase. Please install it using
the following command:

```bash
pip install -e pytorch-lightning/.
```

As next step, we will download the ALFRED data using the script `scripts/download_alfred_data.sh` as follows:

```bash
sh scripts/donwload_alfred_data.sh json_feat
```

Before doing so, make sure that you have installed `p7zip` because is used to extract the trajectory files.

## MaskRCNN fine-tuning

We provide the code to fine-tune a MaskRCNN model on the ALFRED dataset. To create the vision dataset, use the script
`scripts/generate_vision_dataset.sh`. This will create the dataset splits required by the training process. After this,
it's possible to run the model fine-tuning using:

```bash
PYTHONPATH=. python vision/finetune.py --batch_size 8 --gradient_clip_val 5 --lr 3e-4 --gpus 1 --accumulate_grad_batches 2 --num_workers 4 --save_dir storage/models/vision/maskrcnn_bs_16_lr_3e-4_epochs_46_7k_batches --max_epochs 46 --limit_train_batches 7000
```

We provide this code for reference however in our experiments we used the MaskRCNN model from MOCA which applies more
sophisticated data augmentation techniques to improve performance on the ALFRED dataset.

## ALFRED Visual Features extraction

### MaskRCNN

The visual feature extraction script is responsible for generating the MaskRCNN features as well as orientation
information for every bounding box. For the MaskrCNN model, we use the pretrained model from MOCA. You can download it
from their GitHub page. First, we create the directory structure and then download the model weights:

```bash
mkdir -p storage/models/vision/moca_maskrcnn;
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt -O storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt; 
```

We extract visual features for training trajectories using the following command:

```bash
sh scripts/generate_moca_maskrcnn.sh
```

You can refer to the actual extraction script `scripts/generate_maskrcnn_horizon0.py` for additional parameters. We
executed this command on an `p3.2xlarge` instance with NVIDIA V100. This command will populate the directory
`storage/data/alfred/json_feat_2.1.0/` with the visual features for each trajectory step. In particular, the parameter
`--features_folder` will specify the subdirectory (for each trajectory) that will contain the compressed NumPy files
constituting the features. Each NumPy file has the following structure:

```python
dict(
    box_features=np.array,
    roi_angles=np.array,
    boxes=np.array,
    masks=np.array,
    class_probs=np.array,
    class_labels=np.array,
    num_objects=int,
    pano_id=int
)

```

## Data-augmentation procedure

In our paper, we describe a procedure to augment the ALFREd trajectories with object and corresponding receptacle
information. In particular, we reply the trajectories and we make sure to track object and its receptacle during a
subgoal. The data augmentation script will create a new trajectory file called `ref_traj_data.json` that mimics the same
data structure of the original ALFRED dataset but adds to it a few fields for each action.

To start generating the refined data, use the following script:

```bash
PYTHONPATH=. python scripts/generate_landmarks.py 
```

## EmBERT Training

### Vocabulary creation

We use `AllenNLP` for training our models. Before starting the training we will generate the vocabulary for the model
using the following command:

```bash
allennlp build-vocab training_configs/embert/embert_oscar.jsonnet storage/models/embert/vocab.tar.gz --include-package grolp
```

### Training

First, we need to download the OSCAR checkpoint before starting the training process. We used a version of OSCAR which
doesn't use object labels which can be freely downloaded following the instruction
on [GitHub](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md). Make sure to download this file in the
folder `storage/models/pretrained` using the following commands:

```bash
mkdir -p storage/models/pretrained/;
wget https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/base-no-labels.zip -O storage/models/pretrained/oscar.zip;
unzip storage/models/pretrained/oscar.zip -d storage/models/pretrained/;
mv storage/models/pretrained/base-no-labels/ep_67_588997/pytorch_model.bin storage/models/pretrained/oscar-base-no-labels.bin;
rm storage/models/pretrained/oscar.zip;
```

A new model can be trained using the following command:

```bash
allennlp train training_configs/embert/embert_widest.jsonnet -s storage/models/alfred/embert --include-package grolp
```

When training for the first time, make sure to add to the previous command the following parameters:
`--preprocess --num_workers 4`. This will make sure that the dataset is preprocessed and cached in order to speedup
training. We run training using AWS EC2 instances `p3.8xlarge` with `16` workers on a single GPU per configuration.

The configuration file `training_configs/embert/embert_widest.jsonnet` contains all the parameters that you might be
interested in if you want to change the way the model works or any reference to the actual features files. If you're
interested in how to change the model itself, please refer to [the model definition](grolp/models/alfred.py). The
parameters in the constructor of the class will reflect the ones reported in the configuration file. In general, this
project has been developed by using AllenNLP has a reference framework. We refer the reader to the official
[AllenNLP documentation](http://docs.allennlp.org/main/) for more details about how to structure a project.

## EmBERT evaluation

We modified the original ALFRED evaluation script to make sure that the results are completely reproducible. Refer to
the original repository for more information.

To run the evaluation on the `valid_seen` and `valid_unseen` you can use the provided script `scripts/run_eval.sh` in
order to evaluate your model. The EmBERT trainer has different ways of saving checkpoints. At the end of the training,
it will automatically save the best model in an archive named `model.tar.gz` in the destination folder (the one
specified with `-s`). To evaluate it run the following command:

```bash
sh scripts/run_eval.sh <your_model_path>/model.tar.gz 
```

It's also possible to run the evaluation of a specific checkpoint. This can be done by running the previous command as
follows:

```bash
sh scripts/run_eval.sh <your_model_path>/model-epoch=6.ckpt
```

In this way the evaluation script will load the checkpoint at epoch 6 in the path `<your_model_path>`. When specifying a
checkpoint directly, make sure that the folder `<your_model_path>` contains both `config.json` file and `vocabulary`
directory because they are required by the script to load all the correct model parameters.

## Citation

If you're using this codebase please cite our work:

```bibtex
TODO: prepare BibTex when the paper is ready
```