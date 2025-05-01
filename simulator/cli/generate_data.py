import json
import random

# Set a random seed for reproducibility
random.seed(42)

# # Load the existing JSON file
# file_path = "./input_file.json"
# with open(file_path, "r") as file:
#     data = json.load(file)

# # Calculate average input_length and output_length for each step
# step_stats = {}
# for entry in data:
#     for step in entry["Text2SQLRequest"]:
#         step_name = step["step"]
#         if step_name not in step_stats:
#             step_stats[step_name] = {"input_length": [], "output_length": []}
#         step_stats[step_name]["input_length"].append(step["input_length"])
#         step_stats[step_name]["output_length"].append(step["output_length"])

# averages = {
#     step: {
#         "input_length": int(sum(values["input_length"]) / len(values["input_length"])),
#         "output_length": int(sum(values["output_length"]) / len(values["output_length"])),
#     }
#     for step, values in step_stats.items()
# }

# # Generate 50 new entries
# new_entries = []
# for _ in range(100):
#     num_revise = random.randint(0, 10)
#     num_evaluate = random.randint(2, 7)

#     entry = []
#     # Add fixed steps
#     for step in ["Information Retriever", "extract_keywords", "Information Retriever", "Information Retriever", "Information Retriever"]:
#         entry.append({
#             "step": step,
#             "input_length": averages[step]["input_length"],
#             "output_length": averages[step]["output_length"]
#         })
#     # Add generate_candidate_llama-agent steps
#     for _ in range(8):
#         entry.append({
#             "step": "generate_candidate_llama-agent",
#             "input_length": averages["generate_candidate_llama-agent"]["input_length"],
#             "output_length": averages["generate_candidate_llama-agent"]["output_length"]
#         })
#     # Add revise steps
#     for _ in range(num_revise):
#         entry.append({
#             "step": "revise",
#             "input_length": averages["revise"]["input_length"],
#             "output_length": averages["revise"]["output_length"]
#         })
#     # Add unit_tester and generate_unit_test steps
#     entry.append({
#         "step": "unit_tester",
#         "input_length": averages["unit_tester"]["input_length"],
#         "output_length": averages["unit_tester"]["output_length"]
#     })
#     entry.append({
#         "step": "generate_unit_test",
#         "input_length": averages["generate_unit_test"]["input_length"],
#         "output_length": averages["generate_unit_test"]["output_length"]
#     })
#     entry.append({
#         "step": "unit_tester",
#         "input_length": averages["unit_tester"]["input_length"],
#         "output_length": averages["unit_tester"]["output_length"]
#     })
#     # Add evaluate steps
#     for _ in range(num_evaluate):
#         entry.append({
#             "step": "evaluate",
#             "input_length": averages["evaluate"]["input_length"],
#             "output_length": averages["evaluate"]["output_length"]
#         })
#     # Add final unit_tester step
#     entry.append({
#         "step": "unit_tester",
#         "input_length": averages["unit_tester"]["input_length"],
#         "output_length": averages["unit_tester"]["output_length"]
#     })

#     new_entries.append({"Text2SQLRequest": entry})

# # Append the new entries to the existing data
# data.extend(new_entries)

# # Save the updated JSON file
# with open(file_path, "w") as file:
#     json.dump(data, file, indent=4)
file_path = "./input_file_diversed.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Duplicate the "Text2SQLRequest" entries
for entry in data[:55]:  # Only duplicate the first 45 entries
    data.append(entry)

# Save the updated JSON file
with open(file_path, "w") as file:
    json.dump(data, file, indent=4)