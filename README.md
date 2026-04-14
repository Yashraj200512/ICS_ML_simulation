# Deep Learning for Beamforming and Precoding Simulation

This repository contains the simulation code for evaluating deep learning-based approaches for beamforming and hybrid precoding in communication systems. The project integrates MATLAB for channel data generation and Python (TensorFlow/Keras) for designing, training, and testing the neural network models.

## Project Structure

The repository consists of two main components: data generation and neural network simulation.

### 1. Channel Generation & Baselines (MATLAB)
* **`gen_samples.m` / `channel_gen_LOS.m`**: Scripts to generate simulated Channel State Information (CSI) datasets.
* **`HybridPrecoding.m` / `power_allocation.m`**: Traditional baseline algorithms for comparison against the deep learning models.
* **`.mat` files**: The generated perfect and estimated CSI datasets (`pcsi.mat`, `ecsi.mat`) used for training and testing.

### 2. Neural Network Simulation (Python)
* **`train.py` & `train_v2.py`**: Scripts to build and train the neural network models using the generated CSI data.
* **`test.py` & `test_v2.py`**: Scripts to evaluate the trained models on the test datasets.
* **`utils.py` & `utils2.py`**: Helper functions for data loading, processing, and loss calculations.

* **`.h5` files**: Saved weights of the trained models (e.g., `temp_trained.weights.h5`, `20db.h5`).

## Requirements
* **Python 3.x**: `tensorflow`, `numpy`, `scipy`, `matplotlib`
* **MATLAB**: Required only if you wish to generate new CSI datasets or run the traditional baseline comparisons.

## Acknowledgments

The foundational code  was originally developed by [https://github.com/TianLin0509/BF-design-with-DL.git]




