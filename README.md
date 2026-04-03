# EMG-Driven Grip Force Control for Prosthetic Applications

This repository presents a machine learning-based framework for EMG-driven gesture recognition and continuous grip force estimation for prosthetic hand control.

## Overview

Surface electromyography (sEMG) signals are used to enable intuitive prosthetic control by interpreting forearm muscle activity. This work focuses on addressing cross-subject variability using strict Leave-One-Subject-Out (LOSO) evaluation and adaptive calibration.

## Key Contributions

* Strict LOSO evaluation on NinaPro DB2 dataset
* Unified model for gesture recognition and force estimation
* Lightweight subject-specific calibration
* Robustness analysis under fatigue and electrode variability

## Results

* Baseline accuracy: **11.77%**
* Adapted accuracy: **22.04%** (+87.26%)
* Force estimation:

  * MAE: 0.18 N
  * RMSE: 0.24 N
  * R²: 0.91

## Dataset

* NinaPro DB2 (public dataset)

## Paper

The full paper is available here:
📄 `paper/EMG_Grip_Force_Paper.pdf`

## Status

Submitted to an IEEE student conference (under review)

## Future Work

* Real-time implementation
* Cross-dataset validation
* Deep learning models (CNN/LSTM)
