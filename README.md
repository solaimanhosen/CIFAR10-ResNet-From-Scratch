# CIFAR-10 ResNet From Scratch

This project implements and evaluates residual neural networks for CIFAR-10 image classification using PyTorch. It provides two residual architectures: a standard two-convolution residual block (ResNet v1) and a full pre-activation bottleneck block (ResNet v2). Both versions use batch normalization and dropout, and can be trained and tested from the command line.

The implementation was developed as part of COM-S 6730: Advanced Topics in Computational Models of Learning. The complete experimental write-up is available in [HW2_report.pdf](HW2_report.pdf).

## Project overview

CIFAR-10 contains 32 x 32 RGB images from 10 classes. The data pipeline automatically downloads the dataset, converts images to tensors, and normalizes every channel to a mean and standard deviation of 0.5.

The network consists of:

- A 3 x 3 convolutional stem with batch normalization, dropout, and ReLU.
- Three residual stages with 64, 128, and 256 output channels.
- Spatial downsampling at the beginning of stages two and three.
- Adaptive average pooling followed by dropout and a fully connected classifier.
- Standard residual blocks for version 1, with two 3 x 3 convolutions per block.
- Pre-activation bottleneck blocks for version 2, using 1 x 1, 3 x 3, and 1 x 1 convolutions with a channel expansion factor of 4.

Training uses stochastic gradient descent, cross-entropy loss, and momentum of 0.9. CUDA is selected automatically when it is available; otherwise, the model runs on the CPU.

## Evaluated results

The following results are transcribed from the project report. The reported baseline experiments used three residual blocks per stage (`-resnet_size 3`) and trained for 20 epochs.

### Batch normalization

| Metric | Standard block (v1) | Bottleneck block (v2) |
| --- | ---: | ---: |
| Average computation time | 56 s | 54 s |
| Training accuracy | 99.342% | 92.518% |
| Test accuracy | **86.63%** | 79.25% |

The bottleneck model was slightly faster, while the standard block achieved substantially higher training and test accuracy. Both models showed a training-to-test gap of about 13 percentage points, indicating overfitting.

### Batch normalization with dropout

A dropout rate of 0.3 was applied after batch-normalization layers in this experiment.

| Metric | Standard block (v1) | Bottleneck block (v2) |
| --- | ---: | ---: |
| Training accuracy | 88.694% | 67.966% |
| Test accuracy | **85.55%** | 71.20% |

Dropout reduced the standard block's train-test gap from approximately 12.7 to 3.14 percentage points. For the bottleneck model, a dropout rate of 0.3 was too aggressive: its test accuracy fell by roughly eight percentage points, even though its overfitting gap was removed.

### Tuned configuration

The best reported settings for both architectures were a learning rate of 0.01, dropout of 0.1, and 30 training epochs.

| Architecture | Learning rate | Dropout | Epochs | Final test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Standard block (v1) | 0.01 | 0.1 | 30 | **88.91%** |
| Bottleneck block (v2) | 0.01 | 0.1 | 30 | **74.80%** |

The standard-block model produced the best overall result at **88.91% test accuracy**. The report found that the lower dropout rate retained more useful information and that 30 epochs gave the models more time to converge.

These are results from individual training runs, not guaranteed benchmarks. Accuracy and runtime can vary with the random seed, hardware, software versions, and other nondeterministic behavior.

## Parameter-efficiency comparison

The report also compares equal-input/output residual blocks. For a 32-channel standard block, the two 3 x 3 convolutions contain 18,432 weights. A bottleneck block mapping 128 channels through a 32-channel bottleneck contains 17,408 convolution weights. Thus, bottlenecks can add depth while reducing computation and memory through channel compression, although compression may discard useful features and the added depth can make optimization harder.

## Repository structure

| File | Purpose |
| --- | --- |
| `main.py` | Parses command-line arguments and runs training and evaluation. |
| `model.py` | Downloads CIFAR-10, constructs data loaders, and defines training, testing, and seeding utilities. |
| `network.py` | Implements the standard block, bottleneck block, and ResNet model. |
| `requirements.txt` | Lists the Python dependencies. |
| `HW2_report.pdf` | Contains the full assignment report and experimental analysis. |

## Setup

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run requires an internet connection to download CIFAR-10 into `./data`.

## Usage

Run the standard-block model with the report's tuned command-line settings:

```bash
python main.py \
  -resnet_version 1 \
  -resnet_size 3 \
  -num_epochs 30 \
  -batch 16 \
  -lr 0.01 \
  -drop 0.1 \
  -seed 1
```

Use `-resnet_version 2` to select bottleneck blocks. Available arguments are:

| Argument | Default | Description |
| --- | ---: | --- |
| `-seed` | 1 | Random seed. |
| `-resnet_version` | 1 | Residual block type: standard (`1`) or bottleneck (`2`). |
| `-resnet_size` | 18 | Number of residual blocks created in each of the three stages. |
| `-num_classes` | 10 | Number of output classes. |
| `-num_epochs` | 20 | Number of training epochs. |
| `-batch` | 16 | Mini-batch size. |
| `-lr` | 0.01 | SGD learning rate. |
| `-drop` | 0.3 | Final classifier dropout rate. |

In the current implementation, dropout inside residual blocks is fixed at 0.3; `-drop` controls only the final dropout layer before the classifier. Consequently, reproducing the report's fully tuned dropout configuration requires changing the block dropout values in `network.py` as well as passing `-drop 0.1`.
