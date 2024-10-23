import pybamm
import pandas as pd
import numpy as np

def run_simulation_and_save(experiment, parameter_values, filename, nominal_capacity=5):
    """
    Run the battery simulation using PyBaMM for the given experiment and save the results as a CSV.

    Parameters:
    - experiment: PyBaMM experiment object for running the simulation.
    - parameter_values: PyBaMM ParameterValues object.
    - filename: Path to the CSV file where the results will be saved.
    - nominal_capacity: Nominal capacity of the battery in Ah (default: 5 Ah).
    """
    # Load a pre-built model (DFN model for detailed simulations)
    model = pybamm.lithium_ion.DFN()

    # Create a simulation with the specified experiment and parameter values
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)

    # Run the simulation
    solution = sim.solve()

    # Extract the time, voltage, and current from the solution
    time_data = solution["Time [s]"].entries
    voltage = solution["Voltage [V]"].entries
    current = solution["Current [A]"].entries
    temperature = solution["X-averaged cell temperature [K]"].entries  # Convert to Celsius

    # Convert temperature from Kelvin to Celsius
    temperature_c = temperature - 273.15

    # Calculate cumulative capacity (Ah) from the current
    delta_time_hours = np.diff(time_data / 3600, prepend=0)  # Time in hours
    cumulative_capacity = np.cumsum(current * delta_time_hours)

    # Calculate SOC
    soc = 1 - (cumulative_capacity / nominal_capacity)  # SOC is in fraction

    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'Time [s]': time_data,
        'Voltage [V]': voltage,
        'Current [A]': current,
        'SOC': soc,  # State of Charge
        'Temperature [C]': temperature_c
    })

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Simulation results saved to {filename}")


# Define the HPPC and OCV test experiment with grouped steps
experiment = pybamm.Experiment([(
    # HPPC test with proper units
    "Discharge at 1 A for 10 seconds",
    "Rest for 10 minutes",
    "Discharge at 2 A for 10 seconds",
    "Rest for 10 minutes",
    "Charge at 1 A for 10 seconds",
    "Rest for 10 minutes"
), (
    # Grouped OCV test steps
    "Charge at 0.1 A until 4.1 V",
    "Hold at 4.1 V until 50 mA",
    "Rest for 1 hour",
    "Discharge at 0.1 A until 3.0 V",
    "Rest for 1 hour"
)] * 10,  # Repeat for 10 cycles
period="1 second")

# Load the parameter values (use "Chen2020" or another predefined set)
parameter_values = pybamm.ParameterValues("Chen2020")

# Run the simulation and save the results to a CSV (assuming a 5 Ah nominal battery capacity)
run_simulation_and_save(experiment, parameter_values, "battery_simulation_10_cycles.csv", nominal_capacity=5)
