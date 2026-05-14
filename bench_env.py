import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REPO_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_FILE = REPO_DIR / "features.npz"


def format_bytes(value):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def read_meminfo():
    meminfo = {}
    path = Path("/proc/meminfo")
    if not path.exists():
        return meminfo
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, raw = line.split(":", 1)
        value = raw.strip().split()[0]
        try:
            meminfo[key] = int(value) * 1024
        except ValueError:
            pass
    return meminfo


def load_runtime_modules():
    try:
        import numpy as np
        import torch
        import server
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Activate the project environment first, then install dependencies from README."
        ) from exc
    return np, torch, server


def print_system_info(torch, server):
    print("== Environment ==")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor() or 'unknown'}")
    print(f"CPU count: {os.cpu_count()}")
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA: {torch.cuda.is_available()}")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"Torch interop threads: {torch.get_num_interop_threads()}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '-')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', '-')}")
    meminfo = read_meminfo()
    if meminfo:
        print(f"Memory total: {format_bytes(meminfo.get('MemTotal', 0))}")
        print(f"Memory available: {format_bytes(meminfo.get('MemAvailable', 0))}")
    print(f"Features file: {server.FEATURES_FILE}")
    print(f"Model file: {server.MODEL_PATH}")
    print()


def find_sample_images(root, limit):
    root = Path(root)
    if not root.exists():
        return []
    images = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images[:limit]


def load_features(features_file, np, server):
    started = time.perf_counter()
    data = np.load(features_file, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    if "bands" in data:
        bands = server.normalize_feature_bands(names, [str(b) for b in data["bands"]])
    else:
        bands = server.infer_name_bands(names)
    feature_db = data["features"]
    feature_norms = np.linalg.norm(feature_db, axis=1)
    elapsed = time.perf_counter() - started
    return names, bands, feature_db, feature_norms, elapsed


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def print_timing_summary(title, values):
    if not values:
        print(f"{title}: no samples")
        return
    print(
        f"{title}: count={len(values)} "
        f"avg={statistics.mean(values):.3f}s "
        f"median={statistics.median(values):.3f}s "
        f"p95={percentile(values, 95):.3f}s "
        f"min={min(values):.3f}s "
        f"max={max(values):.3f}s"
    )


def bench(args):
    np, torch, server = load_runtime_modules()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    print_system_info(torch, server)

    features_file = Path(args.features)
    if not features_file.exists():
        raise FileNotFoundError(f"features file not found: {features_file}")
    model_file = Path(server.MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"AdaFace model not found: {model_file}")

    print("== Loading ==")
    started = time.perf_counter()
    mtcnn = server.load_mtcnn()
    mtcnn_time = time.perf_counter() - started
    print(f"MTCNN load: {mtcnn_time:.3f}s")

    started = time.perf_counter()
    adaface = server.load_adaface()
    adaface_time = time.perf_counter() - started
    print(f"AdaFace load: {adaface_time:.3f}s")

    names, bands, feature_db, feature_norms, feature_time = load_features(
        features_file, np, server
    )
    print(f"Features load: {feature_time:.3f}s")
    print(f"People: {len(names)}")
    print(f"Feature shape: {feature_db.shape}")
    print()

    sample_images = []
    if args.image:
        sample_images = [Path(args.image)]
    else:
        for root in args.sample_roots:
            sample_images.extend(find_sample_images(root, args.samples - len(sample_images)))
            if len(sample_images) >= args.samples:
                break
    sample_images = sample_images[: args.samples]
    if not sample_images:
        raise FileNotFoundError(
            "no sample image found; pass --image photo.jpg or keep sample images in tests/ or faces/"
        )

    print("== Samples ==")
    for path in sample_images:
        print(f"- {path} ({format_bytes(path.stat().st_size)})")
    print()

    selected_bands = set(args.bands.split(",")) if args.bands else {"mygo", "avemujica", "sumimi"}
    thresholds = server.LOW_MTCNN_THRESHOLDS if args.relaxed else server.DEFAULT_MTCNN_THRESHOLDS
    print("== Recognition Benchmark ==")
    print(f"Rounds: {args.rounds}")
    print(f"Selected bands: {','.join(sorted(selected_bands))}")
    print(f"Thresholds: {thresholds}")

    durations = []
    face_counts = []
    for round_index in range(args.rounds):
        for image_path in sample_images:
            image_bytes = image_path.read_bytes()
            started = time.perf_counter()
            results = server.recognize(
                mtcnn,
                adaface,
                names,
                bands,
                feature_db,
                feature_norms,
                image_bytes,
                thresholds,
                selected_bands,
            )
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            face_counts.append(len(results))
            best = results[0]["name"] if results else "no face"
            print(
                f"round={round_index + 1:02d} image={image_path.name} "
                f"time={elapsed:.3f}s faces={len(results)} best={best}"
            )

    print()
    print_timing_summary("Recognition", durations)
    if durations:
        avg = statistics.mean(durations)
        print(f"Approx throughput: {1 / avg:.2f} req/s at single-worker steady state")
        if avg > 5:
            print("Hint: recognition is slow; try MAX_IMAGE_DIM=960 or fewer Torch threads.")
        elif avg < 2:
            print("Hint: single-worker mode should feel responsive for low traffic.")
    print(f"Total detected faces: {sum(face_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Check environment and benchmark SeiyuuMatch inference.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_FILE))
    parser.add_argument("--image", help="Use one specific image for benchmark.")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample images to auto-pick.")
    parser.add_argument("--rounds", type=int, default=3, help="Benchmark rounds per sample image.")
    parser.add_argument("--bands", default="mygo,avemujica,sumimi", help="Comma-separated recognition bands.")
    parser.add_argument("--relaxed", action="store_true", help="Use relaxed MTCNN thresholds.")
    parser.add_argument("--torch-threads", type=int, help="Override torch.set_num_threads for this run.")
    parser.add_argument(
        "--sample-roots",
        nargs="+",
        default=["tests", "faces"],
        help="Directories used when --image is not set.",
    )
    args = parser.parse_args()
    bench(args)


if __name__ == "__main__":
    main()
