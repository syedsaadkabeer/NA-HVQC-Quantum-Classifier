# Noise-Aware Hybrid Variational Quantum Classifier (NA-HVQC)

## Overview
Implementation of a hybrid quantum-classical classifier using Qiskit, PCA preprocessing, and a softmax classifier. Designed for NISQ-era constraints.

## Features
- Variational Quantum Circuit (VQC)
- Angle Encoding
- Qiskit Aer Simulation
- Hybrid Quantum + Classical Model
- ~90% accuracy on Iris dataset

## Structure
- `src/` → source code  
- `data/` → datasets  
- `figures/` → graphs  
- `results/` → outputs  
- `report/` → final IEEE report  

## Setup
```bash
pip install -r requirements.txt

Run
python src/main.py
Results

Hybrid model achieves ~90% accuracy on Iris dataset.

Authors
Syed Saad Kabeer
Arsalan Mateen
Uzair Ahmed