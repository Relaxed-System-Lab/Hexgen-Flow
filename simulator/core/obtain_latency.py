import os
import sys

decode_time_per_token = {"nvidia_A100": 0.035,
                          "nvidia_A6000": 0.058,
                          "nvidia_L40S": 0.0545}

def obtain_throughput(gpu_type):
    # Map GPU type to the corresponding throughput file
    throughput_files = {
        "nvidia_A100": "../core/Actual_latency/results_A100.txt",
        "nvidia_A6000": "../core/Actual_latency/results_A6000.txt",
        "nvidia_L40S": "../core/Actual_latency/results_L40S.txt"
    }
    file_path = throughput_files.get(gpu_type)
    
    # Read throughput values from the file
    with open(file_path, "r") as f:
        throughput_list = [float(line.strip()) for line in f if line.strip()]
    return throughput_list

def build_latency_dict(gpu_type):
    """
    Build a latency dictionary based on input/output lengths and GPU type.

    Args:
        gpu_type (str): GPU type (A100, A6000, or L40S).

    Returns:
        dict: Latency dictionary.
    """
    latency_dict = {}
    throughput_list = obtain_throughput(gpu_type)
    requests_file = "../core/Actual_latency/requests.txt"

    with open(requests_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or not line.startswith("(") or not line.endswith(")"):
                continue  # Skip empty or improperly formatted lines
            line = line.strip("()").split(",")
            if len(line) != 2:
                continue  # Skip lines that do not have exactly two elements
            input_length = int(line[0].strip())
            output_length = int(line[1].strip())
            latency_dict[(input_length, output_length)] = {}

            # Calculate latencies
            latency = 1 / throughput_list[i]
            per_token_decode_latency = decode_time_per_token[gpu_type]
            prefill_latency = latency - (output_length * per_token_decode_latency)

            latency_dict[(input_length, output_length)]["latency"] = latency
            latency_dict[(input_length, output_length)]["prefill_latency"] = prefill_latency
            latency_dict[(input_length, output_length)]["per_token_decode_latency"] = per_token_decode_latency

    return latency_dict