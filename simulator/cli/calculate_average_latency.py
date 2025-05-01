import json
import os

def calculate_average_latency(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        latencies = data.values()
        return sum(latencies) / len(latencies)

def main():
    base_dir = "./result"
    files = [
        "baseline_output.json",
        "optimized_output0.json",
        "optimized_output1.json",
        # "optimized_output2.json",
        # "optimized_output3.json",
        # "optimized_output4.json",
        # "optimized_output5.json"
    ]
    
    for file_name in files:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            average_latency = calculate_average_latency(file_path)
            print(f"Average latency for {file_name}: {average_latency:.2f} s")
        else:
            print(f"File {file_name} does not exist.")

if __name__ == "__main__":
    main()
