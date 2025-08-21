# iDiffODE: Learning Continuous-Time Latent Dynamics for Generative Spatiotemporal Modeling

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()  
[![Issues](https://img.shields.io/github/issues/yourusername/idiffode.svg)](https://github.com/yourusername/idiffode/issues)  



This repository contains the implementation and evaluation of **iDiffODE**, a hybrid model that combines **Neural ODEs**, **Diffusion Models**, and **Invertible Neural Networks** for effective modeling of irregularly sampled time-series data for the Traffic Accident dataset (https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/).  

---
## Introduction 💡
Irregularly sampled time series are prevalent in domains such as healthcare, wearable sensing, and geospatial event monitoring. These datasets pose significant challenges for traditional time-series models, which assume uniform sampling and struggle with missing or delayed observations. In this work, we present iDiffODE, a novel continuous-time generative model that explicitly constructs smooth latent trajectories from irregular observations, enabling both accurate reconstruction and realistic simulation of multivariate system states. Our approach integrates a Neural Ordinary Differential Equation latent trajectory, which captures time-aware dynamics between arbitrary timestamps, with a diffusion-based prior in latent space to
model multimodal uncertainty. This enables accurate handling of long gaps and asynchronous sampling while generating diverse, data-consistent scenarios. We evaluate our method on three representative datasets, such as PhysioNet ICU records, UCI HAR wearable sensor data, and US national transportation safety accident records, demonstrating consistent improvements over the existing methods. Our results demonstrate that iD-
iffODE and it’s model variants effectively learns continuous latent dynamics and generates time series data even under conditions of severe irregular sampling, positioning it as a highly promising solution for handling irregular time series data in the real world with enhanced generalizability.

---

## Repository structure:
.
├─ README.md
├─ data/
│ └─ raw_combined_accidents.csv # your data
├─ dataset.py # dataset + preprocessing
├─ models.py # MLP / RNN / Transformer model variants + pipeline factory
├─ utils.py # losses, helpers, save/load
├─ train.py # main training script (single entrypoint, uses --model)
├─ run_mlp.py # convenience launcher -> iDiffODE(MLP)
├─ run_rnn.py # convenience launcher -> iDiffODE(RNN)
├─ run_transformer.py # convenience launcher -> iDiffODE(Transformer)
└─ evaluate.py # (optional) evaluation script skeleton

---

## How to use

1.  **Create a new conda environment:**
    ```bash
    conda create -n iDiffODE python=3.9
    conda activate iDiffODE
    ```

2.  **Train MLP variant:**
    ```bash
    python run_mlp.py --csv data/raw_combined_accidents.csv --epochs 200 --bs 32
    ```

3.  **Train RNN variant:**
    ```bash
    python run_rnn.py --csv data/raw_combined_accidents.csv --epochs 200 --bs 32
    ```

4.  **Train Transformer variant:**
    ```bash
    python run_transformer.py --csv data/raw_combined_accidents.csv --epochs 200 --bs 16 --nhead 4 --nlayers 2
    ```

5.  **Evaluate a saved checkpoint:**
    To be uploaded soon [1.1, 1.3.2].

