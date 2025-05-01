import os
import subprocess
import time  # Import the time module

def run_simulations(input_file, n_engines, arrival_rates, gpu_count):
    alpha_values = [round(0.1 * i, 1) for i in range(0, 6)]  # 0, 0.1, ..., 0.5

    for arrival_rate in arrival_rates:
        start_time = time.time()  # Record the start time
        for alpha in alpha_values:
            folder_name = os.path.join("result", f"req_rate={arrival_rate}, alpha={alpha}, gpu={gpu_count}")
            os.makedirs(folder_name, exist_ok=True)

            baseline_output = os.path.join(folder_name, "baseline_output.json")
            optimized_output0 = os.path.join(folder_name, "optimized_output0.json")
            optimized_output1 = os.path.join(folder_name, "optimized_output1.json")

            command = [
                "python", "start_simulator.py",
                "--input", input_file,
                "--n-engines", str(n_engines),
                "--arrival-rate", str(arrival_rate),
                "--alpha", str(alpha),
                "--trace-output", f"./result/trace_output.json",
                "--stats-output", f"./result/stats_output.json"
            ]

            print(f"Running simulation with arrival_rate={arrival_rate}, alpha={alpha}...")
            subprocess.run(command, check=True)

            # # Move only the specified files to the appropriate folder
            # os.rename("./result/baseline_output.json", baseline_output)
            # os.rename("./result/optimized_output0.json", optimized_output0)
            # os.rename("./result/optimized_output1.json", optimized_output1)
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time
        print(f"Total execution time for arrival rate {arrival_rate}: {elapsed_time:.2f} seconds")

if __name__ == "__main__":

    input_file = "./input_file_trace1.json"
    n_engines = 1
    arrival_rates = [0.5, 1.0]  # Iterate over both arrival rates
    gpu_count = 3

    run_simulations(input_file, n_engines, arrival_rates, gpu_count)
