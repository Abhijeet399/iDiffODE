# iDiffODE: Learning Continuous-Time Latent Dynamics for Generative Spatiotemporal Modeling

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()  
[![Issues](https://img.shields.io/github/issues/yourusername/idiffode.svg)](https://github.com/yourusername/idiffode/issues)  



This repository contains the implementation and evaluation of **iDiffODE**, a hybrid model that combines **Neural ODEs**, **Diffusion Models**, and **Invertible Neural Networks** for effective modeling of irregularly sampled time-series data for all the 3 datasets:

1. [Human Activity Recognition dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones/).  
2. [PhysioNet dataset](https://physionet.org/content/challenge-2012/1.0.0/).
3. [Traffic Accident dataset](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/).   

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
├─ requirements.txt
├─ Traffic Accident dataset            # Repository for implementing iDiffODE for the Traffic Accident dataset
├─ PhysioNet dataset                   # Repository for implementing iDiffODE for the PhysioNet dataset
├─ UCI HAR dataset                     # Repository for implementing iDiffODE for the UCI HAR dataset
```
---
## Results

1. [Traffic Accident dataset](https://github.com/Abhijeet399/iDiffODE/blob/main/Traffic%20Accident%20dataset/README.md).     
2. [PhysioNet dataset](https://github.com/Abhijeet399/iDiffODE/blob/main/PhysioNet%20dataset/README.md).
3. [Human Activity Recognition dataset](https://github.com/Abhijeet399/iDiffODE/blob/main/UCI%20HAR%20dataset/README.md).
