from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot


def compress(args):
    recipe = [
        SmoothQuantModifier(smoothing_strength=args.smoothing_strength),
        GPTQModifier(scheme=args.scheme, targets="Linear", ignore=["lm_head"]),
    ]
    # Apply quantization using the built in open_platypus dataset.
    #   * See examples for demos showing how to pass a custom calibration set
    oneshot(
        model=args.model,
        dataset=args.dataset,
        recipe=recipe,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--dataset", type=str, default="open_platypus")
    parser.add_argument(
        "--output_dir", type=str, default="TinyLlama-1.1B-Chat-v1.0-INT8"
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_calibration_samples", type=int, default=512)
    parser.add_argument("--smoothing_strength", type=float, default=0.8)
    parser.add_argument(
        "--scheme", type=str, default="W8A8", choices=["W8A8", "W4A16", "W8A16"]
    )
    args = parser.parse_args()
    print(args)
    compress(args)
