# Picture Version of Task-agnostic Indexes for Deep Learning-based Queries over Unstructured Data

This is the project page for the seminar "A Framework for Measuring the Performance of Selection Queries over Images"
This project is based on the TASTI project and is extended by a additional baseline.

Please read the [paper](https://arxiv.org/abs/2009.04540) for full technical details on the TASTI implementation.

# Requirements

Install the requitements with `pip install -r requirements.txt`. You will also need (via `pip install -e .`):
- [SWAG](https://github.com/stanford-futuredata/swag-python)
- [BlazeIt](https://github.com/stanford-futuredata/blazeit)
- [SUPG](https://github.com/stanford-futuredata/supg)
- Install the tasti package with `pip install -e .` as well.

# Installation
If you want to reproduce our experiments, use Python 3.10 and a conda environment to install `tasti.yml`. You'll also need to install `blazeit`, `supg`, and `tasti` as described below.

Otherwise, the following steps will install the necessary packages from scratch:
```
git clone https://github.com/stanford-futuredata/swag-python.git
cd swag-python/
conda install -c conda-forge opencv
pip install -e .
cd ..

git clone https://github.com/stanford-futuredata/blazeit.git
cd blazeit/
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge pyclipper
pip install -e .
cd ..

git clone https://github.com/stanford-futuredata/supg.git
cd supg/
pip install pandas feather-format
pip install -e .
cd ..

git clone https://github.com/HendrikDroste/picture_tasti.git
cd tasti/
pip install -r requirements.txt
pip install -e .
```

# Reproducing Experiments

We provided the code for our baseline and the image TASTI.
The data for reproducing can be downloaded [here](https://huggingface.co/datasets/imagenet-1k/tree/main/data).
Change the PATH variable in `tasti/examples/picture.py` and `tasti/base_script.py` to update the path in the source code.

The experiments can be reproduced by running the following commands.
Be aware that the embedding and target DNNs has to be switched by editing the code.
```
python tasti/examples/picture.py
python tasti/base_script.py
```

We use [Mask R-CNN ResNet-50 FPN](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection) or [Faster R-CNN ResNet-50 FPN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) as the target dnn.
As emedding DNNs we use different versions of [Resnet](https://pytorch.org/vision/main/models/resnet.html).


# Config
These are the options available in `tasti.IndexConfig` which get passed into the `tasti.Index` object.
- `do_mining`, Boolean that determines whether the mining step is skipped or not
- `do_training`, Boolean that determines whether the training/fine-tuning step of the embedding dnn is skipped or not
- `do_infer`, Boolean that allows you to either compute embeddings or load them from `./cache`
- `do_bucketting`, Boolean that allows you to compute the buckets or load them from `./cache`
- `batch_size`, general batch size for both the target and embedding dnn
- `train_margin`, controls the margin parameter of the triplet loss
- `max_k`, controls the k parameter described in the paper (for computing distance weighted means and votes)
- `nb_train`, controls how many datapoints are labeled to perform the triplet training
- `nb_buckets`, controls the number of buckets used to construct the index
- `nb_training_its`, controls the number of datapoints are passed through the model during training
