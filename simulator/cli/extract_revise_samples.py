import json
from collections import Counter, defaultdict
import random

def count_revise_steps(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    revise_counts = []
    for request_group in data:
        req_lst = request_group["Text2SQLRequest"]
        count = 0
        for request in req_lst:
            if request["step"] == "revise":
                count += 1
        revise_counts.append(count)
    
    distribution = Counter(revise_counts)
    return distribution

def extract_revise_samples(input_filepaths, output_filepath):
    combined_data = []

    # Load and combine data from all input files
    for input_filepath in input_filepaths:
        with open(input_filepath, 'r') as file:
            data = json.load(file)
            combined_data.extend(data)
    
    revise_groups = defaultdict(list)

    # Group "Text2SQLRequest" by the number of "revise" steps
    for request_group in combined_data:
        req_lst = request_group["Text2SQLRequest"]
        count = 0
        for request in req_lst:
            if request["step"] == "revise":
                count += 1
        revise_groups[count].append(request_group)
    
    # Extract 5 samples for each revise count, duplicating if necessary
    sampled_groups = []
    for revise_count, requests in revise_groups.items():
        if len(requests) < 5:
            sampled_groups.extend(requests * (5 // len(requests)) + random.sample(requests, 5 % len(requests)))
        else:
            sampled_groups.extend(random.sample(requests, 5))
    
    # Shuffle the order of the sampled groups
    random.shuffle(sampled_groups)

    # Save the shuffled sampled requests to the output file
    with open(output_filepath, 'w') as file:
        json.dump(sampled_groups, file, indent=4)

if __name__ == "__main__":

    input_filepath = "./input_file_formula1.json"
    input_filepaths = [input_filepath, "./input_file_financial.json"]
    # distribution = count_revise_steps(input_filepath)
    # print("Distribution of 'revise' steps:")
    # for count, frequency in sorted(distribution.items()):
    #     print(f"{count} revise step(s): {frequency} occurrence(s)")

    output_filepath = "./input_file_diversed.json"
    extract_revise_samples(input_filepaths, output_filepath)
    print(f"Sampled requests saved to {output_filepath}")
