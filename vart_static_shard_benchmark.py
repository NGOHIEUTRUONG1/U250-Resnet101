from __future__ import print_function

import argparse
import csv
import gc
import json
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time


def load_runtime_environment(path, xclbin, device_index):
    with open(path) as handle:
        for raw in handle:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            name, value = raw.split("\t", 1)
            os.environ[name] = value
    os.environ["XLNX_VART_FIRMWARE"] = xclbin
    os.environ["XLNX_ENABLE_DEVICES"] = str(device_index)


def find_dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    children = [
        subgraph
        for subgraph in root.toposort_child_subgraph()
        if subgraph.has_attr("device")
        and str(subgraph.get_attr("device")).upper() == "DPU"
    ]
    if len(children) != 1:
        raise RuntimeError("expected one DPU subgraph, got %d" % len(children))
    return children[0]


def prepare_worker(worker_id, dpu_subgraph, warmup):
    import numpy as np
    import vart

    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    inputs = runner.get_input_tensors()
    outputs = runner.get_output_tensors()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("expected one input and one output tensor")
    input_dims = tuple(int(value) for value in inputs[0].dims)
    output_dims = tuple(int(value) for value in outputs[0].dims)
    batch = int(input_dims[0])
    inp = np.zeros(input_dims, dtype=np.int8, order="C")
    out = np.empty(output_dims, dtype=np.int8, order="C")
    for _ in range(warmup):
        job_id = runner.execute_async([inp], [out])
        runner.wait(job_id)
    return {
        "worker": worker_id,
        "runner": runner,
        "input": inp,
        "output": out,
        "batch": batch,
    }


def worker_main(worker, run_id, duration, start_barrier, result_queue):
    try:
        start_barrier.wait()
        start = time.perf_counter()
        deadline = start + duration
        calls = 0
        while time.perf_counter() < deadline:
            job_id = worker["runner"].execute_async(
                [worker["input"]], [worker["output"]]
            )
            worker["runner"].wait(job_id)
            calls += 1
        elapsed = time.perf_counter() - start
        result_queue.put({
            "status": "ok",
            "run": run_id,
            "worker": worker["worker"],
            "pid": os.getpid(),
            "thread_id": threading.current_thread().ident,
            "batch": worker["batch"],
            "calls": calls,
            "frames": calls * worker["batch"],
            "elapsed_seconds": elapsed,
            "fps": calls * worker["batch"] / elapsed,
        })
    except Exception as error:
        result_queue.put({
            "status": "error",
            "run": run_id,
            "worker": worker["worker"],
            "pid": os.getpid(),
            "error": repr(error),
        })


def capture_xrm_snapshot(output_path):
    request = {"request": {"name": "list", "requestId": 901}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(request, handle)
        request_path = handle.name
    try:
        process = subprocess.Popen(
            ["xrmadm", request_path], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True
        )
        text = process.communicate()[0]
        with open(output_path, "w") as handle:
            handle.write(text)
        if process.returncode != 0:
            raise RuntimeError("xrmadm failed with code %d" % process.returncode)
        payload = json.loads(text)
        data = payload["response"]["data"]
        active = []
        for key, device in data.items():
            if not key.startswith("device_") or not isinstance(device, dict):
                continue
            for cu_key, cu in device.items():
                if not cu_key.startswith("cu_") or not isinstance(cu, dict):
                    continue
                normalized = {str(name).strip(): value for name, value in cu.items()}
                channels = int(str(normalized.get("numChanInuse", "0")).strip())
                used_load = str(normalized.get("usedLoad", "0")).split()[0]
                if channels > 0 or int(used_load) > 0:
                    active.append({
                        "device": key,
                        "cu": str(normalized.get("cuName", "")),
                        "channels": channels,
                        "used_load": int(used_load),
                    })
        return active
    finally:
        try:
            os.remove(request_path)
        except OSError:
            pass


def confidence_interval(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    t_value = 2.262 if len(values) == 10 else 1.96
    return mean, sd, t_value * sd / (len(values) ** 0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Static-runner-pool VART throughput benchmark."
    )
    parser.add_argument("--xmodel", required=True)
    parser.add_argument("--fingerprint-env", required=True)
    parser.add_argument("--xclbin", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.workers < 1 or args.repeats < 1 or args.duration <= 0:
        raise ValueError("workers, repeats, and duration must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    load_runtime_environment(
        args.fingerprint_env, args.xclbin, args.device_index
    )

    import xir

    graph = xir.Graph.deserialize(args.xmodel)
    dpu_subgraph = find_dpu_subgraph(graph)
    run_rows = []
    worker_rows = []
    workers = [
        prepare_worker(worker_id, dpu_subgraph, args.warmup)
        for worker_id in range(args.workers)
    ]
    for run_id in range(1, args.repeats + 1):
        result_queue = queue.Queue()
        start_barrier = threading.Barrier(args.workers + 1)
        threads = [
            threading.Thread(
                target=worker_main,
                args=(worker, run_id, args.duration, start_barrier, result_queue),
            )
            for worker in workers
        ]
        for thread in threads:
            thread.start()

        start_barrier.wait()
        time.sleep(min(0.5, args.duration / 4.0))
        snapshot_path = os.path.join(
            args.output_dir, "xrm_active_run_%02d.json" % run_id
        )
        active_cus = capture_xrm_snapshot(snapshot_path)
        active_names = sorted(set(item["cu"] for item in active_cus))
        results = [result_queue.get(timeout=args.duration + 180) for _ in threads]
        for thread in threads:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("worker thread did not exit")
        errors = [row for row in results if row.get("status") != "ok"]
        if errors:
            raise RuntimeError("worker failures: %r" % errors)
        if len(active_names) < args.workers:
            raise RuntimeError(
                "XRM shows %d active CUs for %d workers: %r" %
                (len(active_names), args.workers, active_cus)
            )

        global_elapsed = max(float(row["elapsed_seconds"]) for row in results)
        frames = sum(int(row["frames"]) for row in results)
        fps = frames / global_elapsed
        run_row = {
            "run": run_id,
            "workers": args.workers,
            "frames": frames,
            "seconds": global_elapsed,
            "fps": fps,
            "active_cus": active_names,
        }
        run_rows.append(run_row)
        worker_rows.extend(results)
        print(
            "STATIC_SHARD_RESULT run=%d workers=%d active_cus=%s "
            "FPS=%.6f frames=%d seconds=%.6f" %
            (run_id, args.workers, ",".join(active_names), fps,
             frames, global_elapsed), flush=True
        )
        time.sleep(0.5)

    del workers
    gc.collect()

    fps_values = [float(row["fps"]) for row in run_rows]
    mean, sd, ci = confidence_interval(fps_values)
    summary = {
        "method": "static runner pool; one long-lived VART runner and private buffer pair per worker thread",
        "request_dispatch": "no application-level round-robin",
        "workers": args.workers,
        "repeats": args.repeats,
        "duration_seconds": args.duration,
        "warmup_calls_per_worker": args.warmup,
        "fps_mean": mean,
        "fps_sd": sd,
        "fps_95ci_half_width": ci,
        "fps_min": min(fps_values),
        "fps_max": max(fps_values),
        "runs": run_rows,
        "workers_detail": worker_rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "runs.csv"), "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "workers", "fps", "frames", "seconds", "active_cus"])
        for row in run_rows:
            writer.writerow([
                row["run"], row["workers"], "%.9f" % row["fps"],
                row["frames"], "%.9f" % row["seconds"],
                ";".join(row["active_cus"]),
            ])
    print("STATIC_SHARD_MEAN_FPS=%.9f" % mean)
    print("STATIC_SHARD_SD_FPS=%.9f" % sd)
    print("STATIC_SHARD_95CI_HALF_WIDTH_FPS=%.9f" % ci)
    print("STATIC_SHARD_GATE=PASS")


if __name__ == "__main__":
    main()
