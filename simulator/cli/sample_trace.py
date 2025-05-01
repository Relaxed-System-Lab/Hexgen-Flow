import json
import numpy as np


def sample_trace(input, output, ratio, on):
    print(f"Sampling {input} with ratio {ratio} on column {on}")
    print(f"Loading data from {input}")
    with open(input, "r") as fp:
        data = [json.loads(x) for x in fp.readlines()]
    if on not in data[0]:
        raise ValueError(
            f"Column {on} not found in the input, expected one of {data[0].keys()}"
        )
    column_data = set([x[on] for x in data])
    sampled_column_data = np.random.choice(
        list(column_data), int(len(column_data) * ratio), replace=False
    )
    sampled_data = [x for x in data if x[on] in sampled_column_data]
    print(f"Sampled {len(sampled_data)} out of {len(data)}")
    print(f"Writing sampled data to {output}")
    with open(output, "w") as fp:
        for x in sampled_data:
            fp.write(json.dumps(x) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--ratio", type=float, help="Ratio of the input to be sampled")
    parser.add_argument("--on", type=str, help="Sampling on which column")
    args = parser.parse_args()
    print(args)
    sample_trace(args.input, args.output, args.ratio, args.on)
