# iDiffODE: Learning Continuous-Time Latent Dynamics for Generative Spatiotemporal Modeling


This repository contains the implementation and evaluation of **iDiffODE**, a hybrid model that combines **Neural ODEs**, **Diffusion Models**, and **Invertible Neural Networks** for effective modeling of irregularly sampled time-series data for the Traffic Accident dataset.  

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
│  └─ raw_combined_accidents.csv       # your data
├─ dataset.py                          # dataset + preprocessing
├─ models.py                           # MLP / RNN / Transformer model variants + pipeline factory
├─ utils.py                            # losses, helpers, save/load
├─ train.py                            # main training script (single entrypoint, uses --model)
├─ run_mlp.py                          # convenience launcher -> iDiffODE(MLP)
├─ run_rnn.py                          # convenience launcher -> iDiffODE(RNN)
├─ run_transformer.py                  # convenience launcher -> iDiffODE(Transformer)
└─ evaluate.py                         # (optional) evaluation script skeleton

