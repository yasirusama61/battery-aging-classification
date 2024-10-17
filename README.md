# Battery Aging Classification using PyBAMM

This repository contains code and models for simulating battery aging data using PyBAMM and classifying the aging stages (Early, Mid, End) using machine learning algorithms.

## Project Overview

This project utilizes PyBAMM (Python Battery Mathematical Modeling) to simulate battery data over different charge and discharge cycles. The goal is to predict the aging stage of the battery using features such as capacity, voltage, time, and other factors.

### Key Features:
- **Battery Aging Simulation**: Use PyBAMM to simulate battery behavior over multiple charge-discharge cycles.
- **Data-Driven Classification**: Train machine learning models to classify battery health into aging stages based on simulated data.
- **Real-World Application**: The project is designed to support further integration with real-world battery data from manufacturing or testing environments.

## Folder Structure

- **`data/`**: Contains the generated battery aging data (`battery_aging_data.csv`).
- **`notebooks/`**: Jupyter notebooks for simulation experiments.
- **`scripts/`**: Python scripts for data generation, model training, and evaluation.
- **`models/`**: Pre-trained models (e.g., Random Forest) saved for future use.
- **`requirements.txt`**: Python dependencies required to run the project.
- **`.gitignore`**: Files and folders to ignore in version control.

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your machine. Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
