import json
from collections import defaultdict

# Load the JSON file
file_path = "./input_file_cleaned.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Initialize a dictionary to store step statistics
step_stats = defaultdict(lambda: {"input_length": [], "output_length": []})

# Collect input_length and output_length for each step
for entry in data:
    for step in entry["Text2SQLRequest"]:
        step_name = step["step"]
        step_stats[step_name]["input_length"].append(step["input_length"])
        step_stats[step_name]["output_length"].append(step["output_length"])

# Calculate averages
averages = {
    step: {
        "input_length": sum(values["input_length"]) / len(values["input_length"]),
        "output_length": sum(values["output_length"]) / len(values["output_length"]),
    }
    for step, values in step_stats.items()
}

# Print the averages
for step, stats in averages.items():
    print(f"Step: {step}")
    print(f"  Average Input Length: {stats['input_length']:.2f}")
    print(f"  Average Output Length: {stats['output_length']:.2f}")
