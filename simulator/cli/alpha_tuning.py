from __future__ import annotations

import copy
import itertools
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from scipy import stats as _stats  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _stats = None


@dataclass(frozen=True)
class AutoTuneConfig:
    window_size_s: float = 100.0
    alpha_min: float = 0.0
    alpha_max: float = 0.5
    alpha_step: float = 0.1
    pvalue_threshold: float = 0.01
    initial_window_completions: int = 10
    tuning_workers: int = 1
    tuning_benchmark_max_workers: int = 0
    beta_values: Optional[Sequence[float]] = None


def build_initial_arrival_map(server) -> Dict[str, float]:
    arrivals: Dict[str, float] = {}
    for req in server.pending_requests:
        parent = getattr(req, "parent_request", None)
        if parent is None:
            continue
        req_id = parent.req_id
        if req_id not in arrivals or req.arrive_at < arrivals[req_id]:
            arrivals[req_id] = req.arrive_at
    return arrivals


def compute_p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return math.inf
    sorted_vals = sorted(values)
    index = int(len(sorted_vals) * 0.95)
    if index >= len(sorted_vals):
        index = len(sorted_vals) - 1
    return sorted_vals[index]


def _format_p95(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}s"


def compute_avg(values: Sequence[float]) -> Optional[float]:
    if not values:
        return math.inf
    return statistics.mean(values)


def _format_avg(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}s"


def _normal_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def one_sided_welch_ttest(new_values: Sequence[float], ref_values: Sequence[float]) -> Optional[float]:
    if len(new_values) < 2 or len(ref_values) < 2:
        return None
    mean_new = statistics.mean(new_values)
    mean_ref = statistics.mean(ref_values)
    var_new = statistics.variance(new_values)
    var_ref = statistics.variance(ref_values)
    denom = math.sqrt(var_new / len(new_values) + var_ref / len(ref_values))
    if denom == 0:
        return 0.0 if mean_new > mean_ref else 1.0
    t_stat = (mean_new - mean_ref) / denom
    if t_stat <= 0:
        return 1.0
    if _stats is not None:
        return float(_stats.t.sf(t_stat, df=_welch_df(var_new, var_ref, len(new_values), len(ref_values))))
    return _normal_sf(t_stat)


def _welch_df(var_new: float, var_ref: float, n_new: int, n_ref: int) -> float:
    numerator = (var_new / n_new + var_ref / n_ref) ** 2
    denom = (var_new**2) / (n_new**2 * (n_new - 1)) + (var_ref**2) / (n_ref**2 * (n_ref - 1))
    if denom == 0:
        return float(n_new + n_ref - 2)
    return numerator / denom


def _generate_alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> List[float]:
    if alpha_step <= 0:
        return [alpha_min]
    alphas: List[float] = []
    current = alpha_min
    while current <= alpha_max + 1e-9:
        alphas.append(round(current, 10))
        current += alpha_step
    return alphas


def _seed_completed_ids(server) -> set:
    completed_ids = set()
    for req_id, request in server.text2sql_requests.items():
        if request.current_stage >= request.total_stages:
            completed_ids.add(req_id)
    return completed_ids


def _collect_new_completions(server, arrival_map: Dict[str, float], completed_ids: set) -> List[Tuple[str, float, float]]:
    new_records: List[Tuple[str, float, float]] = []
    for req_id, request in server.text2sql_requests.items():
        if req_id in completed_ids:
            continue
        if request.current_stage >= request.total_stages:
            completed_ids.add(req_id)
            initial_arrival = getattr(request, "arrive_at", None)
            if initial_arrival is None:
                initial_arrival = arrival_map.get(req_id, 0.0)
            completion_time = initial_arrival + request.total_time
            new_records.append((req_id, completion_time, request.total_time))
    return new_records


def _snapshot_server(server, slim_for_tuning: bool = False):
    original_trace = getattr(server, "_trace", None)
    original_text2sql = None
    original_background = None
    original_finished = None
    if original_trace is not None:
        server._trace = []
    if slim_for_tuning:
        original_text2sql = getattr(server, "text2sql_requests", None)
        original_background = getattr(server, "background_requests", None)
        active_parents = set()
        if hasattr(server, "pending_requests"):
            for req in server.pending_requests:
                parent = getattr(req, "parent_request", None)
                if parent is not None:
                    active_parents.add(parent.req_id)
        for model in _sorted_models(server):
            for engine in server.engines[model]:
                for req in itertools.chain(getattr(engine, "waiting", []), getattr(engine, "running", [])):
                    parent = getattr(req, "parent_request", None)
                    if parent is not None:
                        active_parents.add(parent.req_id)
                if hasattr(engine, "finished"):
                    if original_finished is None:
                        original_finished = {}
                    original_finished[engine.engine_id] = engine.finished
                    engine.finished = []
        if original_text2sql is not None:
            server.text2sql_requests = {
                req_id: request
                for req_id, request in original_text2sql.items()
                if request.current_stage < request.total_stages or req_id in active_parents
            }
        if original_background is not None:
            server.background_requests = {
                req_id: request
                for req_id, request in original_background.items()
                if getattr(request, "generation_finished_at", None) is None
            }
    memo = {}
    for model in _sorted_models(server):
        for engine in server.engines[model]:
            analyzer = getattr(engine, "analyzer", None)
            if analyzer is None:
                continue
            config_module = getattr(analyzer, "config", None)
            if config_module is not None:
                memo[id(config_module)] = config_module
    snapshot = copy.deepcopy(server, memo)
    if original_trace is not None:
        server._trace = original_trace
        snapshot._trace = []
    if slim_for_tuning:
        if original_text2sql is not None:
            server.text2sql_requests = original_text2sql
        if original_background is not None:
            server.background_requests = original_background
        if original_finished:
            for model in _sorted_models(server):
                for engine in server.engines[model]:
                    if engine.engine_id in original_finished:
                        engine.finished = original_finished[engine.engine_id]
    return snapshot


def _update_waiting_urgency(server, current_time: float) -> None:
    for model in _sorted_models(server):
        for engine in server.engines[model]:
            engine.update_elapsed_time(current_time, engine.hardware_name)


def _sorted_models(server) -> List[str]:
    return sorted(server.supported_models)


def replay_window(
    snapshot,
    alpha: float,
    window_start: float,
    window_end: float,
    arrival_map: Dict[str, float],
    initial_time_queue: Optional[Sequence[float]] = None,
    beta: Optional[float] = None,
) -> Tuple[Optional[float], int]:
    sim = _snapshot_server(snapshot)
    if sim.global_timer != window_start:
        sim.global_timer = window_start
    _update_waiting_urgency(sim, window_start)
    sim.alpha = alpha
    if beta is not None and hasattr(sim, "policy"):
        try:
            sim.policy.set_beta(beta)
        except Exception:
            sim.policy.beta = float(beta)
    if initial_time_queue is not None:
        time_queue = set(initial_time_queue)
    else:
        time_queue = set()
        for model in _sorted_models(sim):
            for engine in sim.engines[model]:
                time_queue.add(sim.timers[model][engine.engine_id])
    completed_ids = _seed_completed_ids(sim)
    window_latencies: List[float] = []

    while True:
        for model in _sorted_models(sim):
            for engine in sim.engines[model]:
                if sim.timers[model][engine.engine_id] <= sim.global_timer:
                    _event, finished_lst, next_time, _memory_event = engine.step(
                        sim.timers[model][engine.engine_id]
                    )
                    sim.timers[model][engine.engine_id] = next_time
                    if finished_lst:
                        for req in finished_lst:
                            completed_parent = sim.handle_request_completion(req, next_time)
                            if completed_parent is not None:
                                req_id = completed_parent.req_id
                                if req_id not in completed_ids:
                                    completed_ids.add(req_id)
                                    initial_arrival = getattr(completed_parent, "arrive_at", None)
                                    if initial_arrival is None:
                                        initial_arrival = arrival_map.get(req_id, 0.0)
                                    completion_time = initial_arrival + completed_parent.total_time
                                    if window_start <= completion_time < window_end:
                                        window_latencies.append(completed_parent.total_time)
                    time_queue.add(next_time)

        if not sim.has_remaining_requests():
            break
        if not time_queue:
            next_timer = None
            for model in _sorted_models(sim):
                for engine in sim.engines[model]:
                    engine_timer = sim.timers[model][engine.engine_id]
                    if next_timer is None or engine_timer < next_timer:
                        next_timer = engine_timer
            if next_timer is None or next_timer >= window_end:
                break
            sim.global_timer = next_timer
            sim.check_new_requests(sim.global_timer)
            continue

        next_time = min(time_queue)
        if next_time >= window_end:
            break
        sim.global_timer = next_time
        time_queue.remove(next_time)
        sim.check_new_requests(sim.global_timer)

    return compute_p95(window_latencies), len(window_latencies)


def _evaluate_alpha_grid(
    snapshot,
    alpha_grid: Sequence[float],
    window_start: float,
    window_end: float,
    arrival_map: Dict[str, float],
    snapshot_time_queue: Sequence[float],
    workers: int,
) -> Tuple[List[Tuple[float, Optional[float], int]], float]:
    workers = max(1, int(workers))
    start = time.perf_counter()

    if workers == 1:
        replay_results: List[Tuple[float, Optional[float], int]] = []
        for alpha in alpha_grid:
            replay_p95, replay_count = replay_window(
                snapshot,
                alpha,
                window_start,
                window_end,
                arrival_map,
                snapshot_time_queue,
            )
            replay_results.append((alpha, replay_p95, replay_count))
        return replay_results, time.perf_counter() - start

    replay_results = []
    with ThreadPoolExecutor(max_workers=min(workers, len(alpha_grid))) as executor:
        future_map = {
            executor.submit(
                replay_window,
                snapshot,
                alpha,
                window_start,
                window_end,
                arrival_map,
                snapshot_time_queue,
            ): alpha
            for alpha in alpha_grid
        }
        for fut in as_completed(future_map):
            alpha = future_map[fut]
            replay_p95, replay_count = fut.result()
            replay_results.append((alpha, replay_p95, replay_count))

    replay_results.sort(key=lambda x: x[0])
    return replay_results, time.perf_counter() - start


def _evaluate_alpha_beta_grid(
    snapshot,
    alpha_grid: Sequence[float],
    beta_grid: Sequence[float],
    window_start: float,
    window_end: float,
    arrival_map: Dict[str, float],
    snapshot_time_queue: Sequence[float],
    workers: int,
) -> Tuple[List[Tuple[float, float, Optional[float], int]], float]:
    workers = max(1, int(workers))
    start = time.perf_counter()
    combos = list(itertools.product(alpha_grid, beta_grid))

    if workers == 1:
        replay_results: List[Tuple[float, float, Optional[float], int]] = []
        for alpha, beta in combos:
            replay_p95, replay_count = replay_window(
                snapshot,
                alpha,
                window_start,
                window_end,
                arrival_map,
                snapshot_time_queue,
                beta=beta,
            )
            replay_results.append((alpha, beta, replay_p95, replay_count))
        return replay_results, time.perf_counter() - start

    replay_results: List[Tuple[float, float, Optional[float], int]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(combos))) as executor:
        future_map = {
            executor.submit(
                replay_window,
                snapshot,
                alpha,
                window_start,
                window_end,
                arrival_map,
                snapshot_time_queue,
                beta=beta,
            ): (alpha, beta)
            for (alpha, beta) in combos
        }
        for fut in as_completed(future_map):
            alpha, beta = future_map[fut]
            replay_p95, replay_count = fut.result()
            replay_results.append((alpha, beta, replay_p95, replay_count))

    replay_results.sort(key=lambda x: (x[0], x[1]))
    return replay_results, time.perf_counter() - start


def run_auto_tuned_simulation(server, config: AutoTuneConfig) -> None:
    arrival_map = build_initial_arrival_map(server)
    window_start = 0.0
    window_size_s: Optional[float] = None
    window_end: Optional[float] = None
    prev_window_latencies: Optional[List[float]] = None
    pending_records: List[Tuple[float, float]] = []
    all_latencies: List[float] = []
    initial_target = max(0, config.initial_window_completions)
    initial_window_defined = initial_target == 0
    if initial_window_defined:
        window_size_s = config.window_size_s
        window_end = window_start + window_size_s
    completed_ids = _seed_completed_ids(server)
    time_queue = set()
    snapshot = _snapshot_server(server, slim_for_tuning=True)
    snapshot_time_queue = set(time_queue)
    alpha_schedule: List[Tuple[float, float]] = [(window_start, server.alpha)]
    beta_schedule: Optional[List[Tuple[float, float]]] = None
    if config.beta_values is not None:
        current_beta = getattr(getattr(server, "policy", None), "beta", 1.0)
        beta_schedule = [(window_start, float(current_beta))]
    window_count = 0
    tuning_trigger_count = 0
    tuning_event_durations: List[Tuple[float, float, float]] = []
    did_benchmark = False

    print("Auto-tune enabled.")
    if _stats is None:
        print("Note: scipy not found, using normal approximation for t-test p-values.")

    while True:
        for model in _sorted_models(server):
            for engine in server.engines[model]:
                if server.timers[model][engine.engine_id] <= server.global_timer:
                    event, finished_lst, next_time, memory_event = engine.step(
                        server.timers[model][engine.engine_id]
                    )
                    server.timers[model][engine.engine_id] = next_time
                    if finished_lst:
                        for req in finished_lst:
                            completed_parent = server.handle_request_completion(req, next_time)
                            if completed_parent is not None:
                                req_id = completed_parent.req_id
                                if req_id not in completed_ids:
                                    completed_ids.add(req_id)
                                    initial_arrival = getattr(completed_parent, "arrive_at", None)
                                    if initial_arrival is None:
                                        initial_arrival = arrival_map.get(req_id, 0.0)
                                    completion_time = initial_arrival + completed_parent.total_time
                                    pending_records.append((completion_time, completed_parent.total_time))
                                    all_latencies.append(completed_parent.total_time)
                    time_queue.add(next_time)
                    if event is not None:
                        server._trace.extend(event)
                        server._trace.append(memory_event)

        if not initial_window_defined and len(pending_records) >= initial_target:
            sorted_records = sorted(pending_records, key=lambda x: x[0])
            cutoff_time = sorted_records[initial_target - 1][0]
            window_end = cutoff_time + 1e-9
            window_size_s = window_end - window_start
            if server.global_timer < window_end:
                server.global_timer = window_end
            _update_waiting_urgency(server, window_end)
            window_latencies = [lat for ct, lat in pending_records if ct < window_end]
            pending_records = [(ct, lat) for ct, lat in pending_records if ct >= window_end]
            window_p95 = compute_p95(window_latencies)
            window_avg = compute_avg(window_latencies)
            completed_count = len(window_latencies)
            window_count += 1
            print(
                f"[Window {window_start:.2f}-{window_end:.2f}) completed={completed_count}, "
                f"Avg={_format_avg(window_avg)}, P95={_format_p95(window_p95)}"
            )
            print(
                f"Initial window size set to {window_size_s:.2f}s based on first {initial_target} completions."
            )
            prev_window_latencies = window_latencies
            initial_window_defined = True
            window_start = window_end
            window_end = window_start + window_size_s
            snapshot = _snapshot_server(server, slim_for_tuning=True)
            snapshot_time_queue = set(time_queue)

        if time_queue:
            next_time = min(time_queue)
            if window_end is not None and next_time >= window_end:
                if server.global_timer < window_end:
                    server.global_timer = window_end
                _update_waiting_urgency(server, window_end)
                window_latencies = [
                    lat for ct, lat in pending_records if window_start <= ct < window_end
                ]
                pending_records = [(ct, lat) for ct, lat in pending_records if ct >= window_end]
                window_p95 = compute_p95(window_latencies)
                window_avg = compute_avg(window_latencies)
                completed_count = len(window_latencies)
                window_count += 1
                print(
                    f"[Window {window_start:.2f}-{window_end:.2f}) completed={completed_count}, "
                    f"Avg={_format_avg(window_avg)}, P95={_format_p95(window_p95)}"
                )

                trigger_tuning = False
                p_value = None
                prev_p95 = compute_p95(prev_window_latencies or [])
                prev_avg = compute_avg(prev_window_latencies or [])
                if completed_count == 0:
                    print(
                        f"No completions in window {window_start:.2f}-{window_end:.2f}; skipping tuning."
                    )
                elif prev_window_latencies:
                    p_value = one_sided_welch_ttest(window_latencies, prev_window_latencies)
                    if p_value is not None and window_p95 > prev_p95 and p_value < config.pvalue_threshold:
                        trigger_tuning = True

                if trigger_tuning:
                    tuning_trigger_count += 1
                    print(
                        f"Tuning triggered for window {window_start:.2f}-{window_end:.2f}: "
                        f"prev P95={_format_p95(prev_p95)}, curr P95={_format_p95(window_p95)}, "
                        f"prev Avg={_format_avg(prev_avg)}, curr Avg={_format_avg(window_avg)}, p={p_value}"
                    )
                    alpha_grid = _generate_alpha_grid(
                        config.alpha_min, config.alpha_max, config.alpha_step
                    )
                    beta_grid = list(config.beta_values) if config.beta_values is not None else None

                    if beta_grid is not None:
                        replay_results, tuning_duration = _evaluate_alpha_beta_grid(
                            snapshot,
                            alpha_grid,
                            beta_grid,
                            window_start,
                            window_end,
                            arrival_map,
                            list(snapshot_time_queue),
                            config.tuning_workers,
                        )
                    else:
                        replay_results, tuning_duration = _evaluate_alpha_grid(
                            snapshot,
                            alpha_grid,
                            window_start,
                            window_end,
                            arrival_map,
                            list(snapshot_time_queue),
                            config.tuning_workers,
                        )
                    tuning_event_durations.append((window_start, window_end, tuning_duration))
                    print(
                        f"Auto-tuning wall time for this window (workers={config.tuning_workers}): {tuning_duration:.4f}s"
                    )

                    if config.tuning_benchmark_max_workers and not did_benchmark and beta_grid is None:
                        max_workers = max(1, min(int(config.tuning_benchmark_max_workers), 5))
                        print(f"Tuning benchmark (same window) for workers=1..{max_workers}:")
                        for w in range(1, max_workers + 1):
                            _bench_results, bench_time = _evaluate_alpha_grid(
                                snapshot,
                                alpha_grid,
                                window_start,
                                window_end,
                                arrival_map,
                                list(snapshot_time_queue),
                                w,
                            )
                            print(f"  workers={w} -> {bench_time:.4f}s")
                        did_benchmark = True

                    if beta_grid is None:
                        replay_results = [
                            (
                                alpha,
                                (math.inf if (completed_count > 0 and cnt == 0) else p95),
                                cnt,
                            )
                            for (alpha, p95, cnt) in replay_results
                        ]
                        for alpha, replay_p95, replay_count in replay_results:
                            if completed_count > 0 and replay_count == 0 and alpha == server.alpha:
                                print(
                                    "Warning: replay with current alpha produced no completions; snapshot may be inconsistent."
                                )
                        replay_results.sort(
                            key=lambda x: (float("inf") if x[1] is None else x[1], x[0])
                        )
                        finite_results = [r for r in replay_results if r[1] is not None and not math.isinf(r[1])]
                        if finite_results:
                            chosen_alpha = min(finite_results, key=lambda x: (x[1], x[0]))[0]
                        else:
                            chosen_alpha = server.alpha
                        if chosen_alpha != server.alpha:
                            server.alpha = chosen_alpha
                            alpha_schedule.append((window_end, chosen_alpha))
                        print("Replay P95 by alpha:")
                        for alpha, replay_p95, replay_count in replay_results:
                            p95_str = _format_p95(replay_p95)
                            print(f"  alpha={alpha:.3f} -> P95={p95_str} (n={replay_count})")
                        print(
                            f"Alpha updated at t={window_end:.2f}s to {server.alpha:.3f}"
                        )
                    else:
                        replay_results = [
                            (
                                alpha,
                                beta,
                                (math.inf if (completed_count > 0 and cnt == 0) else p95),
                                cnt,
                            )
                            for (alpha, beta, p95, cnt) in replay_results
                        ]
                        replay_results.sort(
                            key=lambda x: (float("inf") if x[2] is None else x[2], x[0], x[1])
                        )
                        finite_results = [r for r in replay_results if r[2] is not None and not math.isinf(r[2])]
                        if finite_results:
                            chosen_alpha, chosen_beta = min(
                                finite_results, key=lambda x: (x[2], x[0], x[1])
                            )[:2]
                        else:
                            chosen_alpha = server.alpha
                            chosen_beta = getattr(getattr(server, "policy", None), "beta", 1.0)
                        if chosen_alpha != server.alpha:
                            server.alpha = chosen_alpha
                            alpha_schedule.append((window_end, chosen_alpha))
                        if hasattr(server, "policy"):
                            try:
                                server.policy.set_beta(chosen_beta)
                            except Exception:
                                server.policy.beta = float(chosen_beta)
                        if beta_schedule is not None:
                            beta_schedule.append((window_end, float(chosen_beta)))
                        print("Replay P95 by alpha/beta:")
                        for alpha, beta, replay_p95, replay_count in replay_results:
                            p95_str = _format_p95(replay_p95)
                            print(
                                f"  alpha={alpha:.3f}, beta={beta:.3f} -> P95={p95_str} (n={replay_count})"
                            )
                        print(
                            f"Alpha/beta updated at t={window_end:.2f}s to {server.alpha:.3f}/{chosen_beta:.3f}"
                        )

                prev_window_latencies = window_latencies
                window_start = window_end
                window_end = window_start + (window_size_s or config.window_size_s)
                snapshot = _snapshot_server(server, slim_for_tuning=True)
                snapshot_time_queue = set(time_queue)

            next_time = min(time_queue)
            server.global_timer = next_time
            time_queue.remove(next_time)
            server.check_new_requests(server.global_timer)

        if not server.has_remaining_requests():
            break

    if alpha_schedule:
        print("Alpha schedule:")
        for time_s, alpha in alpha_schedule:
            print(f"  t={time_s:.2f}s -> alpha={alpha:.3f}")
    if beta_schedule is not None:
        print("Beta schedule:")
        for time_s, beta in beta_schedule:
            print(f"  t={time_s:.2f}s -> beta={beta:.3f}")
    if initial_window_defined and pending_records:
        final_end = max(ct for ct, _ in pending_records) + 1e-9
        final_latencies = [lat for ct, lat in pending_records if ct >= window_start]
        final_avg = compute_avg(final_latencies)
        final_p95 = compute_p95(final_latencies)
        window_count += 1
        print(
            f"[Final Window {window_start:.2f}-{final_end:.2f}) completed={len(final_latencies)}, "
            f"Avg={_format_avg(final_avg)}, P95={_format_p95(final_p95)}"
        )
    if all_latencies:
        overall_p95 = compute_p95(all_latencies)
        overall_avg = compute_avg(all_latencies)
        print(
            f"Overall end-to-end Avg across all completed queries: {_format_avg(overall_avg)} (n={len(all_latencies)})"
        )
        print(
            f"Overall end-to-end P95 across all completed queries: {_format_p95(overall_p95)} (n={len(all_latencies)})"
        )

    if window_count > 0:
        print(
            f"Auto-tuning triggering frequency: {tuning_trigger_count}/{window_count} "
            f"({(tuning_trigger_count / window_count) * 100:.2f}%)"
        )
    if tuning_event_durations:
        one = tuning_event_durations[0]
        print(
            f"Example tuning event wall time: window {one[0]:.2f}-{one[1]:.2f} took {one[2]:.4f}s "
            f"(workers={config.tuning_workers})"
        )
