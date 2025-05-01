from simulator.internal.analyzer.model_analyzer import ModelAnalyzer


def analyze_perf(
    model_id,
    hardware_name: str,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    w_bit: int,
    a_bit: int,
    kv_bit=16,
):
    analyzer = ModelAnalyzer(
        model_id=model_id,
        hardware=hardware_name,
        config_file="adaml/internal/configs/llama.py",
        source="huggingface",
    )
    gen_results = analyzer.analyze_generate_task(
        prompt_len=prompt_len,
        gen_len=gen_len,
        batchsize=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=True,
    )
    print(gen_results)
    return {
        "prefill": gen_results["prefill_time"],
        "decode": gen_results["inference_time"] - gen_results["prefill_time"],
        "inference_time": gen_results["inference_time"],
    }


if __name__ == "__main__":
    results = analyze_perf(
        "meta-llama/Llama-2-7b-hf", "nvidia_A6000", 8, 2048, 128, 4, 4, 16
    )
    prefill_time = results["prefill_time"]
    decode_time = results["inference_time"] - prefill_time
    print(f"Prefill Time: {prefill_time}")
    print(f"Decode Time: {decode_time}")
