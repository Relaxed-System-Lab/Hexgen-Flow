import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import json
import math
from dataclasses import asdict
from simulator.core.global_engine import LLMGlobalEngine
from simulator.core.global_engine_optimized import OPGlobalEngine
from simulator.core.utils import load_trace
from simulator.ui import make_table
from rich.console import Console
from huggingface_hub import login
import matplotlib.pyplot as plt  # Add this import for plotting
from simulator.core.request import STANDARD_WORKFLOW, calculate_avg_empirical_time
from simulator.cli.alpha_tuning import AutoTuneConfig, compute_avg, compute_p95, run_auto_tuned_simulation


console = Console()

def _format_latency(value):
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}s"


def _print_overall_end_to_end(server):
    latencies = [
        request.total_time
        for request in server.text2sql_requests.values()
        if request.current_stage >= request.total_stages
    ]
    avg_latency = compute_avg(latencies)
    p95_latency = compute_p95(latencies)
    print(
        f"Overall end-to-end Avg across all completed queries: {_format_latency(avg_latency)} "
        f"(n={len(latencies)})"
    )
    print(
        f"Overall end-to-end P95 across all completed queries: {_format_latency(p95_latency)} "
        f"(n={len(latencies)})"
    )


def _print_completion_p95(server):
    total = len(server.text2sql_requests)
    completed_latencies = [
        request.total_time
        for request in server.text2sql_requests.values()
        if request.current_stage >= request.total_stages
    ]
    completed = len(completed_latencies)
    completion_rate = (completed / total) * 100 if total else 0.0
    p95_latency = compute_p95(completed_latencies)
    print(f"Completion rate: {completion_rate:.2f}% ({completed}/{total})")
    print(f"P95 latency for completed queries: {_format_latency(p95_latency)} (n={completed})")

def run_simulation(args, mode):
    # Create server with specified mode
    server = LLMGlobalEngine(mode)

    for i in range(args.n_engines):
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_A6000", 4,4,4)
        server.add_engine("meta-llama/Llama-3.1-70B-Instruct", "nvidia_L40S", 4,4,4)
    
    server.load_requests(
        args.input,
        float(args.arrival_rate),
        args.SLO,
        args.multi_tenant,
        getattr(args, "arrival_shift", False),
    )
    
    server.start()
    if mode == "baseline":
        server.save_results("./result/baseline_output.json")
    elif mode == "ray":
        server.save_results("./result/ray_output.json")
    elif mode == "llf":
        server.save_results("./result/llf_output.json")
    elif mode == "ray-llf":
        server.save_results("./result/ray_llf_output.json")
    elif mode == "vtc":
        server.save_results("./result/vtc_output.json")
    elif mode == "rr+pq":
        server.save_results("./result/rr+pq_output.json")
    elif mode == "qlm":
        server.save_results("./result/qlm_output.json")

    # Trace output disabled.
    # with open(args.trace_output, "w") as f:
    #     data = {"traceEvents": [asdict(x) for x in server.trace]}
    #     f.write(json.dumps(data, indent=4))
    stats = {
        "summary": server.requests_stats,
        # "failed": server.failed_requests,
        "config": server.config,
    }
    # Stats output disabled.
    # with open(args.stats_output, "w") as f:
    #     f.write(json.dumps(stats, indent=4))

    print(end="\n")
    print(f"--" * 10 + " Simulation Done " + "--" * 10)

    console.print(make_table("Summary", server.summary))
    if mode == "ray-llf":
        _print_completion_p95(server)
    
    if args.multi_tenant:
        # Show tenant-specific results
        from simulator.core.request import calculate_tenant_slo
        tenant1_slo = calculate_tenant_slo(server.hardware_lst, 0)
        tenant2_slo = calculate_tenant_slo(server.hardware_lst, 1)
        tenant1_pass_rate, tenant2_pass_rate = server.multi_tenant_SLO_pass_rate(tenant1_slo, tenant2_slo)
        print(f"\nMulti-tenant Results:")
        print(f"Tenant 1 (Base SLO): {tenant1_slo:.2f}s - Pass Rate: {tenant1_pass_rate:.2%}")
        print(f"Tenant 2 (1.5x SLO): {tenant2_slo:.2f}s - Pass Rate: {tenant2_pass_rate:.2%}")
    
    # print(f"Pass rate: {server.SLO_pass_rate(float(args.SLO))}")
    slo_scales = [round(x, 2) for x in [3 + 0.2 * i for i in range(60)]]  # 0.3 to 2.0
    pass_rates = [server.SLO_pass_rate(args.SLO * scale) for scale in slo_scales]
    return pass_rates

def run_simulation_optimized(args, w1=1, index=0, mode="hexflow"):
    # print(args)
    server = OPGlobalEngine(alpha=args.alpha, mode=mode)
    if hasattr(server, "policy") and getattr(args, "beta", None) is not None:
        try:
            server.policy.set_beta(args.beta)
        except Exception:
            server.policy.beta = float(args.beta)

    for i in range(args.n_engines):
        server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_A100", 4,4,4)
        server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_A6000", 4,4,4)
        server.add_engine(w1, "meta-llama/Llama-3.1-70B-Instruct", "nvidia_L40S", 4,4,4)

    server.load_requests(
        args.input,
        float(args.arrival_rate),
        args.SLO,
        args.multi_tenant,
        getattr(args, "arrival_shift", False),
    )

    if getattr(args, "auto_tune", False):
        beta_values = None
        if getattr(args, "tune_beta", False):
            raw = getattr(args, "beta_values", "")
            if raw:
                beta_values = [float(x) for x in raw.split(",") if x.strip() != ""]
            if not beta_values:
                beta_values = [0.5, 1.0, 2.0]
        config = AutoTuneConfig(
            window_size_s=args.window_size,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_step=args.alpha_step,
            initial_window_completions=args.initial_window_completions,
            tuning_workers=args.tuning_workers,
            tuning_benchmark_max_workers=args.tuning_benchmark_max_workers,
            beta_values=beta_values,
        )
        run_auto_tuned_simulation(server, config)
    else:
        server.start()
        _print_overall_end_to_end(server)
    if mode == "vtc":
        server.save_results("./result/vtc_output.json")
    elif mode == "qlm":
        server.save_results("./result/qlm_output.json")
    else:
        server.save_results(f"./result/optimized_output{index}.json")

    # Trace output disabled.
    # with open(f"./result/optimized_trace_output{index}.json", "w") as f:
    #     data = {"traceEvents": [asdict(x) for x in server.trace]}
    #     f.write(json.dumps(data, indent=4))
    stats = {
        "summary": server.requests_stats,
        # "failed": server.failed_requests,
        "config": server.config,
    }
    # Stats output disabled.
    # with open(f"./result/optimized_stats_output{index}.json", "w") as f:
    #     f.write(json.dumps(stats, indent=4))

    print(end="\n")
    print(f"--" * 10 + " Simulation Done " + "--" * 10)

    console.print(make_table("Summary", server.summary))
    
    if args.multi_tenant:
        # Show tenant-specific results
        from simulator.core.request import calculate_tenant_slo
        tenant1_slo = calculate_tenant_slo(server.hardware_lst, 0)
        tenant2_slo = calculate_tenant_slo(server.hardware_lst, 1)
        tenant1_pass_rate, tenant2_pass_rate = server.multi_tenant_SLO_pass_rate(tenant1_slo, tenant2_slo)
        print(f"\nMulti-tenant Results:")
        print(f"Tenant 1 (Base SLO): {tenant1_slo:.2f}s - Pass Rate: {tenant1_pass_rate:.2%}")
        print(f"Tenant 2 (1.5x SLO): {tenant2_slo:.2f}s - Pass Rate: {tenant2_pass_rate:.2%}")

    slo_scales = [round(x, 2) for x in [3 + 0.2 * i for i in range(60)]]  # 0.3 to 2.0
    pass_rates = [server.SLO_pass_rate(args.SLO * scale) for scale in slo_scales]
    return pass_rates

if __name__ == "__main__":
    import argparse
    # hardware_lst = ["nvidia_A100", "nvidia_A100", "nvidia_A100", "nvidia_A100"]
    # hardware_lst = ["nvidia_A100", "nvidia_A100", "nvidia_A6000", "nvidia_A6000"]
    hardware_lst = ["nvidia_A100", "nvidia_A100", "nvidia_A6000", "nvidia_L40S"]
    slo = sum([calculate_avg_empirical_time(hardware_lst, s) for s in STANDARD_WORKFLOW])
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--n-engines", type=int, help="Number of engines")
    parser.add_argument("--arrival-rate", help="Arrival rate", default=None)
    parser.add_argument("--SLO", help="Text2SQL Request SLO", default=slo)
    parser.add_argument("--alpha", type=float, help="Alpha value for the optimized engine", default=0.0)
    parser.add_argument("--beta", type=float, help="Beta multiplier for workload ratio baseline", default=1.0)
    parser.add_argument("--multi-tenant", action="store_true", help="Enable multi-tenant mode with different SLOs")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "baseline", "vtc", "qlm", "hexflow", "optimized", "ray", "llf", "ray-llf"],
        help="Simulation mode. Use 'hexflow'/'optimized' for the alpha-tunable engine.",
    )
    parser.add_argument(
        "--arrival-shift",
        action="store_true",
        help="Use shifted arrivals (3s@5qps, 4s@20qps, 3s@5qps) instead of Poisson.",
    )
    parser.add_argument("--auto-tune", action="store_true", help="Enable alpha auto-tuning (hexflow only)")
    parser.add_argument("--window-size", type=float, default=100.0, help="Auto-tune window size (seconds)")
    parser.add_argument("--alpha-min", type=float, default=0.0, help="Auto-tune alpha grid min")
    parser.add_argument("--alpha-max", type=float, default=0.5, help="Auto-tune alpha grid max")
    parser.add_argument("--alpha-step", type=float, default=0.1, help="Auto-tune alpha grid step")
    parser.add_argument(
        "--tune-beta",
        action="store_true",
        help="If set, auto-tune will search beta values alongside alpha (hexflow only).",
    )
    parser.add_argument(
        "--beta-values",
        type=str,
        default="",
        help="Comma-separated beta values for auto-tune (e.g., 0.5,1,2).",
    )
    parser.add_argument("--initial-window-completions", type=int, default=5, help="Number of completed Text2SQL queries to set the initial window size")
    parser.add_argument(
        "--tuning-workers",
        type=int,
        default=1,
        help="Number of parallel workers for alpha replay during a tuning event",
    )
    parser.add_argument(
        "--tuning-benchmark-max-workers",
        type=int,
        default=0,
        help="If >0, benchmark the first tuning event with workers=1..min(this,5) and log wall time",
    )

    parser.add_argument(
        "--trace-output",
        type=str,
        help="Trace file",
        default="./result/trace_output.json",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        help="Stats file",
        default="./result/stats_output.json",
    )
    args = parser.parse_args()
    # slo_scales = [round(x, 2) for x in [3 + 0.2 * i for i in range(60)]]  # 0.3 to 3.2
    
    if args.mode in ("hexflow", "optimized"):
        run_simulation_optimized(args, 1, 1, "hexflow")
    elif args.mode == "baseline":
        run_simulation(args, "baseline")
    elif args.mode == "ray":
        run_simulation(args, "ray")
    elif args.mode == "llf":
        run_simulation(args, "llf")
    elif args.mode == "ray-llf":
        run_simulation(args, "ray-llf")
    elif args.mode == "vtc":
        run_simulation(args, "vtc")
    elif args.mode == "qlm":
        run_simulation(args, "qlm")
    else:
        pass_rates_vllm = run_simulation(args, "baseline")
        pass_rates_vtc = run_simulation(args, "vtc")
        pass_rate_qlm = run_simulation(args, "qlm")
        _ = (pass_rates_vllm, pass_rates_vtc, pass_rate_qlm)
    # pass_rates_vtc = run_simulation_optimized(args, 0, 0, "vtc")
    # pass_rate_qlm = run_simulation_optimized(args, 0, 0, "qlm")
    
    # pass_rates_rrpq = run_simulation(args, "rr+pq")
    # pass_rates_wbfcfs = run_simulation_optimized(args, 0, 0)
    # pass_rates_wbpq = run_simulation_optimized(args, 1, 1)
    # # Plotting the pass rates for both baseline and optimized
    # plt.figure(figsize=(10, 6))
    # # plt.plot(slo_scales, pass_rates_rrpq, marker='o', label="RR+PQ SLO Pass Rate", color='blue')
    # plt.plot(slo_scales, pass_rates_vtc, marker='o', label=" VTC SLO Pass Rate", color='blue')
    # plt.plot(slo_scales, pass_rates_wbfcfs, marker='o', label="WB+FCFS SLO Pass Rate", color='red')
    # plt.plot(slo_scales, pass_rates_wbpq, marker='o', label="WB+PQ SLO Pass Rate", color='green')
    # plt.title(f"SLO Pass Rate vs SLO Scale")
    # plt.xlabel("SLO Scale")
    # plt.ylabel("Pass Rate")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(f"./result/slo_pass_plot_1.png")
    # plt.close()
