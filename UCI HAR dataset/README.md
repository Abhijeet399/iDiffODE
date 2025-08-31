# iDiffODE: Learning Continuous-Time Latent Dynamics for Generative Spatiotemporal Modeling

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()  
[![Issues](https://img.shields.io/github/issues/yourusername/idiffode.svg)](https://github.com/yourusername/idiffode/issues)  



This repository contains the implementation and evaluation of **iDiffODE**, a hybrid model that combines **Neural ODEs**, **Diffusion Models**, and **Invertible Neural Networks** for effective modeling of irregularly sampled time-series data for the [Human Activity Recognition dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones/).  

---
## Introduction 💡
Irregularly sampled time series are prevalent in domains such as healthcare, wearable sensing, and geospatial event monitoring. These datasets pose significant challenges for traditional time-series models, which assume uniform sampling and struggle with missing or delayed observations. In this work, we present iDiffODE, a novel continuous-time generative model that explicitly constructs smooth latent trajectories from irregular observations, enabling both accurate reconstruction and realistic simulation of multivariate system states. Our approach integrates a Neural Ordinary Differential Equation latent trajectory, which captures time-aware dynamics between arbitrary timestamps, with a diffusion-based prior in latent space to
model multimodal uncertainty. This enables accurate handling of long gaps and asynchronous sampling while generating diverse, data-consistent scenarios. We evaluate our method on three representative datasets, such as PhysioNet ICU records, UCI HAR wearable sensor data, and US national transportation safety accident records, demonstrating consistent improvements over the existing methods. Our results demonstrate that iDiffODE and it’s model variants effectively learns continuous latent dynamics and generates time series data even under conditions of severe irregular sampling, positioning it as a highly promising solution for handling irregular time series data in the real world with enhanced generalizability.

![Alt text describing the image](https://i.postimg.cc/k51xpZzX/iDiffODE.png)

---

## Repository structure:
```text
.
├─ README.md
├─ data_dir/
│  └─ train       # your data
│  └─ test        # your data
├─ dataset.py                          # dataset + preprocessing
├─ models.py                           # MLP / RNN / Transformer model variants + pipeline factory
├─ utils.py                            # losses, helpers, save/load
├─ train.py                            # main training script (single entrypoint, uses --model)
├─ run_mlp.py                          # convenience launcher -> iDiffODE(MLP)
├─ run_rnn.py                          # convenience launcher -> iDiffODE(RNN)
├─ run_transformer.py                  # convenience launcher -> iDiffODE(Transformer)
└─ evaluate.py                         # To be uploaded soon

```
---

## How to use

1.  **Create a new conda environment:**
    ```bash
    conda create -n iDiffODE python=3.9
    conda activate iDiffODE
    ```

2.  **Train MLP variant:**
    ```bash
    python run_mlp.py --data_dir /path/to/UCI_HAR --epochs 200 --batch_size 32 --lr 1e-3
    ```

3.  **Train RNN variant:**
    ```bash
    python run_rnn.py --data_dir /path/to/UCI_HAR --epochs 200
    ```

4.  **Train Transformer variant:**
    ```bash
    python run_transformer.py --data_dir /path/to/UCI_HAR --epochs 200 --hidden_dim 256 --latent_dim 128
    ```

5.  **Evaluate a saved checkpoint:**

    To be uploaded soon [1.1, 1.3.2].
6. **If you want to Resume training from a checkpoint:**
    ```bash
    python train.py --data_dir /path/to/UCI_HAR --resume checkpoints/mlp_epoch10.pth
    ````
---

## Results

### Quantitative Results
COMPREHENSIVE MODEL PERFORMANCE COMPARISON. ↑ INDICATES HIGHER IS BETTER ; ↓ INDICATES LOWER IS BETTER . BOLD INDICATES BEST PERFORMANCE . HERE IDIFFODE_M IS OUR MODEL PROPOSED USING MLP IN NEURAL ODE, IDIFFODE_R IS OUR MODEL PROPOSED USING RNN IN NEURAL ODE, IDIFFODE_T IS OUR MODEL PROPOSED USING TRANSFORMERS IN NEURAL ODE.
![Alt text describing the image](https://i.postimg.cc/D0FqcJj6/Screenshot-from-2025-08-31-02-00-37.png)


COMPARATIVE ANALYSIS OF THREE ARCHITECTURAL VARIANTS OF OUR PROPOSED FRAMEWORK IDIFFODE_M, IDIFFODE_R AND IDIFFODE_T EVALUATED ON THREE DIVERSE , IRREGULARLY SAMPLED TIME - SERIES DATASETS : TRAFFIC ACCIDENT, PHYSIO_NET, AND UCI_HAR.
![Alt text describing the image](https://i.postimg.cc/66mWnJ82/Screenshot-from-2025-08-31-02-04-30.png)


### Qualitative Resutls

This figure represents the Heatmap of the origional and the reconstructed feature vectors:
![Alt text describing the image](https://i.postimg.cc/tJYVk3dh/HeatMap.png)


This figure represents the Feature-wise value comparision of the origional and the reconstructed feature vectors:
![Alt text describing the image](https://i.postimg.cc/gcQsxsPy/Featurewise-value.png)

This figure represents the Feature-wise absolute error bar chart of the origional and the reconstructed feature vectors:
![Alt text describing the image](https://i.postimg.cc/XvB7YnKk/Featurewise-error.png)

This figure represents the Radar plot of the origional and the reconstructed feature vectors:
![Alt text describing the image](https://i.postimg.cc/QtLNBv0G/radar-plot.png)


