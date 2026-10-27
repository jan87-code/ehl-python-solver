"""CLI report for hardware detection and adaptive backend selection."""

import argparse
import json

from ehl_solver.hardware import (
    detect_hardware,
    estimate_convolution_memory,
    report_as_dict,
    select_backend,
    selection_as_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect EHL hardware, validate optional backends, and select a strategy."
    )
    parser.add_argument("--mode", choices=("auto", "cpu", "hip", "torch"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "hip", "cuda"), default="auto")
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        metavar=("NY", "NX"),
        default=(129, 129),
        help="pressure grid shape in (ny, nx) order",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    if min(args.shape) < 1:
        parser.error("shape dimensions must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = detect_hardware()
    try:
        selection = select_backend(
            args.mode,
            args.device,
            tuple(args.shape),
            repeats=args.repeats,
            report=report,
        )
    except (RuntimeError, ValueError) as exc:
        error_data = {
            "mode": args.mode,
            "selection_error": str(exc),
            "memory_estimate": estimate_convolution_memory(tuple(args.shape)).__dict__,
            "hardware": report_as_dict(report),
        }
        if args.json:
            print(json.dumps(error_data, indent=2, sort_keys=True))
        else:
            print(f"Selection failed: {exc}")
            for name, status in report.backends.items():
                print(
                    f"{name}: installed={status.installed}, "
                    f"operational={status.operational}, reason={status.reason or '-'}"
                )
        raise SystemExit(2) from exc
    data = selection_as_dict(selection)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return

    hardware = data["hardware"]
    print(f"System: {hardware['system']} {hardware['machine']}")
    print(f"Processor: {hardware['processor'] or 'unknown'}")
    print(f"CPU instruction flags: {', '.join(hardware['cpu_flags']) or 'unavailable'}")
    print("Backend validation:")
    for name, status in hardware["backends"].items():
        state = "operational" if status["operational"] else "unavailable"
        print(
            f"  {name}: installed={status['installed']}, {state}, "
            f"device={status['device'] or '-'}, arch={status['architecture'] or '-'}, "
            f"float64={status['float64']}, fft={status['fft']}, "
            f"cpu_operational={status['cpu_operational']}"
        )
        if status["reason"]:
            print(f"    {status['reason']}")
    estimate = data["memory_estimate"]
    print(
        f"Workload: {estimate['grid_shape']} grid, padded convolution "
        f"{estimate['padded_shape']}, HIP estimate {estimate['hip_bytes']} B, "
        f"PyTorch estimate {estimate['torch_bytes']} B"
    )
    print(
        f"Selected: {data['selected_backend']} on {data['selected_device']} "
        f"({data['strategy']})"
    )
    if data["benchmarks"]:
        print("Measured median times (device / end-to-end ms):")
        for benchmark in data["benchmarks"]:
            cached = " cached" if benchmark["cached"] else ""
            print(
                f"  {benchmark['backend']}: {benchmark['device_time_ms']:.3f} / "
                f"{benchmark['end_to_end_ms']:.3f}{cached}"
            )


if __name__ == "__main__":
    main()
