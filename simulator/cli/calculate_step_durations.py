import json
from collections import defaultdict

# Load the JSON file
with open('./result/stats_output.json', 'r') as file:
    data = json.load(file)

# Initialize a dictionary to store total durations and counts for each step
step_durations = defaultdict(lambda: {"total_duration": 0, "count": 0})

# Iterate through the summary records
for record in data["summary"]:
    step = record["step"]
    duration = record["generation_finished_at"] - record["arrive_at"]
    step_durations[step]["total_duration"] += duration
    step_durations[step]["count"] += 1

# Calculate the average duration for each step
average_durations = {
    step: durations["total_duration"] / durations["count"]
    for step, durations in step_durations.items()
}

# Print the results
print("Average durations for each step:")
for step, avg_duration in average_durations.items():
    print(f"{step}: {avg_duration:.4f}")
