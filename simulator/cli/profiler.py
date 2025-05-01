from simulator.profiler.profiler import profile
from simulator.internal.analyzer import ModelAnalyzer

def main(args):
    analyzer = ModelAnalyzer(args.model_id, args.hardware, "simulator/internal/configs/llama.py", "huggingface")
    profile(analyzer, args.seq_len, args.batch_size, args.w_bit, args.a_bit, args.kv_bit)
    


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Model ID", default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--hardware", type=str, help="Hardware", default='nvidia_A100')
    parser.add_argument("--seq_len", type=int, help="Sequence length", default=1024)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--w_bit', type=int, help='Weight bit', default=16)
    parser.add_argument('--a_bit', type=int, help='Activation bit', default=16)
    parser.add_argument('--kv_bit', type=int, help='Key/Value bit', default=16)
    args = parser.parse_args()
    main(args)