import json
from pathlib import Path

def extract_unique_pairs(filepaths):
    unique_pairs = set()
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            data = json.load(file)
            for entry in data:
                requests = entry.get("Text2SQLRequest", [])
                for request in requests:
                    input_length = request.get("input_length")
                    output_length = request.get("output_length")
                    if input_length is not None and output_length is not None:
                        unique_pairs.add((input_length, output_length))
    return unique_pairs

def main():
    filepaths = [
        "./input_file_trace1.json",
        "./input_file_trace2.json",
        "./input_file_trace3.json"
    ]
    unique_pairs = extract_unique_pairs(filepaths)
    output_file = "./unique_pairs_output.txt"
    with open(output_file, 'w') as file:
        file.write("Unique (input_length, output_length) pairs:\n")
        for pair in sorted(unique_pairs):
            file.write(f"{pair}\n")

if __name__ == "__main__":
    main()
