import json
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from simulator.core.request import STANDARD_WORKFLOW, calculate_avg_empirical_time

hardware_lst = ["nvidia_A100", "nvidia_A100", "nvidia_A6000", "nvidia_L40S"]
SLO = sum([calculate_avg_empirical_time(hardware_lst, s) for s in STANDARD_WORKFLOW])
# print(f"SLO: {SLO}")
# SLO = 142.13
SCALES = [round(x, 2) for x in [4 + 0.1 * i for i in range(60)]]  # 0.3 to 3.2

def calculate_pass_rate(file_path, scale, num_requests=50):
    """
    Calculate the pass rate for the first `num_requests` requests in the given JSON file
    based on the threshold defined by `scale * SLO`.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "r") as f:
        data = json.load(f)

    threshold = scale * SLO
    values = list(data.values())[25:75]
    passed_requests = sum(1 for value in values if value <= threshold)
    pass_rate = passed_requests / num_requests
    return pass_rate

def main():
    # File paths
    files = [
        "./result/optimized_output0.json",
        "./result/optimized_output1.json",
        "./result/baseline_output.json"
    ]

    labels = ["Optimized Output 0", "Optimized Output 1", "Baseline Output"]
    colors = ["red", "green", "blue"]  # Assign different colors for each label
    all_pass_rates = []

    # Calculate pass rates for each file and scale
    for file_path in files:
        pass_rates = []
        for scale in SCALES:
            pass_rate = calculate_pass_rate(file_path, scale)
            if pass_rate is not None:
                pass_rates.append(pass_rate)
        all_pass_rates.append(pass_rates)

    # Plotting
    plt.figure(figsize=(10, 6))
    for pass_rates, label, color in zip(all_pass_rates, labels, colors):
        plt.plot(SCALES, pass_rates, marker='o', label=label, color=color)

    plt.title("Pass Rate vs SLO Scale for the Middle 50 Requests")
    plt.xlabel("SLO Scale")
    plt.ylabel("Pass Rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.savefig("./result/pass_rate_vs_slo_scale.png")
    plt.show()

if __name__ == "__main__":
    main()
