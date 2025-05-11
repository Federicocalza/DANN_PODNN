# DANN_PODNN

An extension of the Domain-Adversarial Neural Network (DANN) model with PODNN-style domain discriminators and optional orthogonality regularization between classifier branches.

## 📌 Description

This repository provides code for training a DANN model with the following improvements:
- Multi-branch PODNN-style domain discriminator.
- Optional orthogonality regularization between domain classifier branches.
- Support for both source-only and adversarial (DANN) training modes.

## 🗂️ Datasets

### Supported datasets (`--dset`)
| Code    | Source/Target Dataset         |
|---------|-------------------------------|
| `s` or `sv` | SVHN                     |
| `m`     | MNIST                         |
| `u`     | USPS                          |
| `mm`    | MNIST-M (included in `./data`)|
| `sd`    | Synthetic Digits (manual)     |
| `signs` | Synthetic Signs ↔ GTRSB       |

> ✅ **MNIST**, **USPS**, and **SVHN** are automatically downloaded by the script.  
> ❗ **Synthetic Digits**, **Signs**, and **GTRSB** must be manually placed in the `./data/` folder.
> ❗ **Possible adaptation:**
	>s2m SVHN → MNIST
	>m2s MNIST → SVHN
	>m2u  MNIST → USPS
	>u2m USPS → MNIST
	>m2mm MNIST → MNIST-M
	>sv2sd SVHN → Synthetic Digits
	>signs Syn Signs → GTRSB


The **training set** is unlabeled, while the **test set** is labeled.

## ⚙️ Command-line options

| Argument             | Type    | Default       | Description |
|----------------------|---------|----------------|-------------|
| `--method`           | str     | `src`          | Choose between `src` (source-only training) or `dann` (adversarial training). If `dann` is selected and no source model exists, it will be trained first. |
| `--src_epochs`       | int     | `50`           | Number of epochs for source-only training. |
| `--adapt_epochs`     | int     | `350`          | Number of epochs for DANN training. |
| `--batch_size`       | int     | `128`          | Batch size for DANN training (as used in the original paper by Ganin et al.). |
| `--lr`               | float   | `1e-4`         | Learning rate. |
| `--weight_decay`     | float   | `1e-5`         | Weight decay for Adam optimizer. |
| `--dset`             | str     | `s2m`          | Dataset pair (e.g., `s2m` means SVHN → MNIST). |
| `--data_path`        | str     | `./data/`      | Path to datasets. |
| `--model_path`       | str     | `./model/`     | Path to save/load models. If not specified, a folder named after `--dset` will be used. |
| `--seed`             | int     | `100`          | Random seed for reproducibility. |
| `--num_branches`     | int     | `10`           | Number of branches in the domain discriminator. |
| `--PODNN_stride`     | int     | `10`           | Stride used for aggregating the PODNN branches. **Highly sensitive parameter.** |
| `--mode`             | str     | `test`         | Choose whether to `train` or `test` the model. |
| `--ortho_scaling`    | float   | `0.5`          | Not currently active. Can be used to scale orthogonality regularization. |
| `--debug`            | bool    | `False`        | Enables debug mode in case of runtime errors. |


## 🚀 Usage examples

### Train a source-only model
```bash
python main.py --method src --dset u2m --mode train
```

### Train a DANN model with PODNN discriminator
```bash
python main.py --method dann --dset s2m --num_branches 15 --PODNN_stride 5 --mode train
```

### Test a trained model
```bash
python main.py --dset s2m --mode test
```

## 📁 Project structure

```
DANN_PODNN/
├── data/                # Datasets
├── model/               # Saved model checkpoints
├── utils/               # Utility functions
├── main.py              # Main script
├── README.md            # This file
```

## 📋 License

Distributed under the MIT License.

## 🙋‍♂️ Contact

Created by [Federico Calza](https://github.com/Federicocalza)
federicocalza.1095@gmail.com  
For questions, feedback, or collaborations: feel free to open an issue or reach out!

## Credits
- [Original DANN Implementation] (https://github.com/s-chh/PyTorch-DANN) for providing the original DANN code and model
- [PODNN Framework] (https://github.com/caisr-hh/podnn/tree/master/podnn) for providing the tools for the PODNN pipeline 
