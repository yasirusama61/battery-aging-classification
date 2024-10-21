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
```
### Experiment Results

For this experiment, we used the **Single Particle Model (SPM)** in PyBAMM and ran a simulation for 10 cycles using the following conditions:
- **Discharge** at 0.4C until 2.5 V
- **Rest** for 10 minutes
- **Charge** at 0.5C until 4.2 V
- **Hold** at 4.2 V until current drops to 50 mA
- **Rest** for 10 minutes after charging

#### Observations:
- **Voltage and Current Patterns**: Both voltage and current showed consistent behavior across the 10 cycles. The voltage varied between 2.5V and 4.2V during charge-discharge cycles, and the current reversed between positive and negative values, indicating proper charge and discharge phases.
- **Rest Periods**: The battery showed expected behavior during the rest periods, with the current dropping to zero.
- **Initial Aging Test**: We initially tested for aging over 10 cycles and observed no significant signs of degradation. The voltage and capacity remained stable across the cycles, with no major capacity fade or voltage sag. It is expected that more extensive cycling (e.g., 100 or more cycles) will reveal gradual aging effects such as capacity fade or an increase in internal resistance.

### Plots:
Below are key plots that were generated from the 10-cycle experiment:

1. **Voltage and Current vs. Time**: This plot shows the voltage and current behavior during the 10 charge-discharge cycles. The consistent pattern indicates stable battery behavior across all cycles.

   ![Voltage and Current over Time](voltage_current_plot.png)

### Insights:
- This experiment highlights the stability of the battery over the initial 10 cycles, with no visible degradation.
- Further testing with extended cycles will be necessary to observe aging trends and to classify the battery into different aging stages (Early, Mid, End).