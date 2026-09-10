"""
High-Performance Benchmark & Stress-Testing Suite for CCCD Pipeline.
Measures CPU, GPU Utilization, System RAM, GPU VRAM, Latency Percentiles (P50/P90/P95/P99), and Throughput (Req/s).
"""

import argparse
import os
from pathlib import Path
import time
from typing import List
import cv2
import numpy as np
import psutil
import torch

from src.pipeline.cccd_pipeline import CCCDDetectionPipeline
from src.utils.logger import get_logger

logger = get_logger("BenchmarkPerformance")


def get_vram_usage_mb() -> float:
    """Get current GPU VRAM allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_max_vram_usage_mb() -> float:
    """Get peak GPU VRAM allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def get_ram_usage_mb() -> float:
    """Get process physical RAM memory (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def run_benchmark(
    source: str,
    total_requests: int = 1000,
    batch_size: int = 8,
    fp16: bool = True,
    device: str = "",
    unclip_ratio: float = 1.75,
):
    logger.info("=" * 65)
    logger.info("CCCD PIPELINE SERVER STRESS TEST & BENCHMARK")
    logger.info("=" * 65)

    # 1. Load image into RAM
    src_p = Path(source)
    if not src_p.exists():
        raise FileNotFoundError(f"Source image not found: {source}")

    img_bgr = cv2.imread(str(src_p))
    if img_bgr is None:
        raise ValueError(f"Could not decode image from: {source}")

    h, w = img_bgr.shape[:2]
    logger.info(f"Input Image: {src_p.name} | Resolution: {w}x{h}")
    logger.info(f"Target Requests: {total_requests:,} | Batch Size: {batch_size} | FP16: {fp16}")

    # 2. Initialize pipeline
    ram_before = get_ram_usage_mb()
    vram_before = get_vram_usage_mb()

    logger.info(f"Initial RAM: {ram_before:.1f} MB | Initial VRAM: {vram_before:.1f} MB")
    logger.info("Initializing models on device...")

    pipeline = CCCDDetectionPipeline(
        device=device,
        fp16=fp16,
        dbnet_unclip_ratio=unclip_ratio,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 3. Warmup
    logger.info("Executing GPU / Model Warmup (5 runs)...")
    pipeline.warmup(num_runs=5)

    ram_after_init = get_ram_usage_mb()
    vram_after_init = get_vram_usage_mb()
    logger.info(f"Model Resident RAM: {ram_after_init:.1f} MB (+{ram_after_init - ram_before:.1f} MB)")
    logger.info(f"Model Resident VRAM: {vram_after_init:.1f} MB (+{vram_after_init - vram_before:.1f} MB)")

    # 4. Stress Test Execution
    logger.info(f"Starting Stress Test with {total_requests:,} requests...")

    batch_images = [img_bgr] * batch_size
    num_batches = (total_requests + batch_size - 1) // batch_size
    latencies: List[float] = []

    cpu_percentages = []
    t_global_start = time.perf_counter()

    for b_idx in range(num_batches):
        t0 = time.perf_counter()
        
        if batch_size == 1:
            _ = pipeline.predict(img_bgr)
            batch_count = 1
        else:
            _ = pipeline.predict_batch(batch_images, batch_size=batch_size)
            batch_count = len(batch_images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_req_ms = elapsed_ms / batch_count
        for _ in range(batch_count):
            latencies.append(per_req_ms)

        if b_idx % max(1, (num_batches // 10)) == 0 or b_idx == num_batches - 1:
            cpu_percentages.append(psutil.cpu_percent(interval=None))
            completed = min((b_idx + 1) * batch_size, total_requests)
            logger.info(
                f"Progress: [{completed:6d}/{total_requests:6d}] ({(completed/total_requests)*100:5.1f}%) | "
                f"Batch Latency: {elapsed_ms:6.1f} ms | "
                f"Per-Req: {per_req_ms:5.2f} ms"
            )

    t_total_s = time.perf_counter() - t_global_start
    actual_requests = len(latencies)

    # 5. Compute Metrics
    latencies = latencies[:total_requests]
    latencies.sort()
    
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    fps = total_requests / t_total_s

    peak_ram = get_ram_usage_mb()
    peak_vram = get_max_vram_usage_mb()
    avg_cpu = np.mean(cpu_percentages) if cpu_percentages else psutil.cpu_percent()

    # 6. Report Summary
    print("\n" + "=" * 65)
    print("                BENCHMARK & HARDWARE METRICS SUMMARY")
    print("=" * 65)
    print(f"Total Requests Executed:    {total_requests:,}")
    print(f"Total Time Elapsed:         {t_total_s:.3f} seconds")
    print(f"Throughput (Requests/sec):  {fps:.2f} FPS / Req/s")
    print("-" * 65)
    print("LATENCY BREAKDOWN (per image):")
    print(f"  - Min Latency:            {min_latency:6.2f} ms")
    print(f"  - Average Latency:        {avg_latency:6.2f} ms")
    print(f"  - P50  (Median):          {p50:6.2f} ms")
    print(f"  - P90  Percentile:        {p90:6.2f} ms")
    print(f"  - P95  Percentile:        {p95:6.2f} ms")
    print(f"  - P99  Percentile:        {p99:6.2f} ms")
    print(f"  - Max Latency:            {max_latency:6.2f} ms")
    print("-" * 65)
    print("HARDWARE RESOURCE CONSUMPTION:")
    print(f"  - Average CPU Utilization: {avg_cpu:5.1f} %")
    print(f"  - Peak System RAM:        {peak_ram:6.1f} MB (~{peak_ram/1024:.2f} GB)")
    if torch.cuda.is_available():
        print(f"  - Peak GPU VRAM (CUDA):   {peak_vram:6.1f} MB (~{peak_vram/1024:.2f} GB)")
        print(f"  - GPU Device Name:        {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"  - Acceleration Backend:   Apple Silicon Metal (MPS Unified Memory)")
    else:
        print(f"  - Acceleration Backend:   CPU")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CCCD Pipeline Server Benchmark & Stress Test")
    parser.add_argument("--source", type=str, default="data/thidong_F.png", help="Path to test image")
    parser.add_argument("--requests", type=int, default=1000, help="Total number of requests to simulate")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 half precision")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16")
    parser.add_argument("--device", type=str, default="", help="Device (cuda, mps, cpu)")
    parser.add_argument("--unclip-ratio", type=float, default=1.75, help="Vatti unclip expansion ratio")
    args = parser.parse_args()

    run_benchmark(
        source=args.source,
        total_requests=args.requests,
        batch_size=args.batch_size,
        fp16=args.fp16,
        device=args.device,
        unclip_ratio=args.unclip_ratio,
    )


if __name__ == "__main__":
    main()
