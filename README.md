# Hexgen-Flow

This is a simulator designed to evaluate scheduling algorithms proposed by Hexgen-Flow when serving Text-to-SQL queries. These queries typically involve multiple LLM (Large Language Model) inference calls. The simulator allows users to experiment with different configurations, including input traces, arrival rates, and GPU deployment settings, to analyze the performance of the scheduling algorithm.

## Features

- **Simulate Text-to-SQL Queries**: The simulator processes input trace files containing LLM inference requests and simulates their execution.
- **Customizable Configurations**: Users can modify input traces, arrival rates, and GPU settings to explore various scenarios.
- **Heterogeneous GPU Deployment**: Supports different GPU types and configurations.
- **Performance Metrics**: Reports latency for each end-to-end Text-to-SQL query and provides detailed statistics.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

Install the dependencies using:

```bash
pip install -r simulator/requirements.txt
```

### Running the Simulator

The main script to run the simulator is `start_simulator.py`. Below is an example command:

```bash
python simulator/cli/start_simulator.py \
    --input ./simulator/cli/input_file_trace3.json \
    --n-engines 4 \
    --arrival-rate 1.0 \
    --trace-output ./result/trace_output.json \
    --stats-output ./result/stats_output.json
```

#### Command-Line Arguments

- `--input`: Path to the input trace file (JSON format).
- `--n-engines`: Number of model instances.
- `--arrival-rate`: Arrival rate of Text-to-SQL queries.
- `--trace-output`: Path to save the trace output file.
- `--stats-output`: Path to save the statistics output file.
- `--SLO`: (Optional) Service Level Objective for Text-to-SQL requests.
- `--alpha`: (Optional) Alpha value for the optimized engine.

### Input Trace File

The input trace file should be a JSON file containing the input and output lengths of each LLM inference request. Example input files are provided in the `simulator/cli/` directory.

### Output

- **Trace Output**: Contains detailed trace events of the simulation.
- **Latency Output**: Contains the latency results for the simulation, saved as `latency{index}.json`. This file provides latency metrics for each end-to-end Text-to-SQL query processed under the scheduling algorithm.

## Customization

Users can modify the `start_simulator.py` script to:

- Experiment with different GPU configurations.
- Implement new scheduling algorithms.
- Adjust the number of engines or other parameters.

## Example Use Cases

1. **Analyze Different Arrival Rates**:
   Change the `--arrival-rate` parameter to observe how the system performs under varying query loads.

2. **Test Different Input Traces**:
   Use different input trace files with the `--input` parameter to simulate various Text-to-SQL query patterns.

3. **Experiment with GPU Settings**:
   Modify the `--n-engines` parameter and the GPU configurations in `start_simulator.py` to test heterogeneous GPU deployments.

## Repository Structure

- `simulator/`: Contains the core simulation logic and utilities.
  - `cli/`: Command-line interface scripts.
  - `core/`: Core simulation components, including engines and policies.
  - `internal/`: Internal utilities and configurations.
  - `profiler/`: Profiling tools.
  - `ui/`: User interface utilities for displaying results.

