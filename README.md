# Deep Learning for Beamforming and Precoding

## Overview
This repository implements a Deep Learning-based approach for beamforming and hybrid precoding in Massive MIMO communication systems. The primary objective is to use neural networks to approximate the performance of traditional, non-convex mathematical optimization algorithms (such as Orthogonal Matching Pursuit) for calculating optimal precoding matrices. 

By shifting the heavy computational load to a trained neural network, the system can predict highly accurate precoders significantly faster during real-world deployment, maximizing the overall Sum Rate (Spectral Efficiency).

## Project Architecture
The complete system pipeline involves simulating physical channel data, training the network, and evaluating the final predictions. 

**This repository contains the finalized Inference and Evaluation pipeline:**
* **Channel Data:** Utilizes pre-simulated Estimated Channel State Information (`ecsi.mat`) and Perfect Channel State Information (`pcsi.mat`), representing complex multi-path wireless environments.
* **Deep Learning Inference:** Deploys a pre-trained Keras/TensorFlow model (`20db.h5`, `temp_trained.weights.h5`) to process the incoming channel data and output predicted precoders.
* **Benchmarking:** Evaluates the neural network's predictions directly against the theoretical limits established by traditional mathematical baselines.

## Repository Structure
* **`test.py`**: The core execution script that loads the test data and passes it through the pre-trained neural network to measure performance.
* **`utils.py`**: Essential helper functions for loading, normalizing, and formatting the MATLAB `.mat` dataset files for Python.
* **`ecsi.mat` & `pcsi.mat`**: The testing datasets containing the thousands of simulated physical channel matrices.
* **`20db.h5` & `temp_trained.weights.h5`**: The saved weights of the trained neural network models.

## Requirements
* Python 3.x
* TensorFlow / Keras
* NumPy
* SciPy

## How to Run the Simulation
To evaluate the model's performance on the test dataset and generate the predictions, run the following command in your terminal:

```bash
python test.py