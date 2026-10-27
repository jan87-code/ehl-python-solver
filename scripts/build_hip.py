"""Build the optional native HIP extension using the installed ROCm SDK."""

import argparse
import os
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=os.environ.get("ROCM_PATH"))
    parser.add_argument("--arch", default="gfx1032")
    args = parser.parse_args()

    hipcc = shutil.which("hipcc")
    if hipcc is None:
        parser.error("hipcc was not found on PATH; activate the ROCm SDK environment first")
    sdk_candidates = []
    if args.sdk:
        sdk_candidates.append(Path(args.sdk).resolve())
    sdk_candidates.append(Path(sysconfig.get_paths()["purelib"]) / "_rocm_sdk_devel")
    sdk_venv = Path(hipcc).resolve().parent.parent
    sdk_candidates.append(
        sdk_venv / "lib" / f"python{sysconfig.get_python_version()}" /
        "site-packages" / "_rocm_sdk_devel"
    )
    sdk_candidates.append(sdk_venv)
    sdk = next(
        (candidate for candidate in sdk_candidates
         if (candidate / "include/hipfft/hipfft.h").is_file()
         and (candidate / "lib/libhipfft.so").exists()),
        None,
    )
    if sdk is None:
        parser.error(
            "Could not locate ROCm SDK include/ and lib/ directories; pass --sdk "
            "with the _rocm_sdk_devel directory"
        )
    include_dir = Path(sysconfig.get_paths()["include"])
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        parser.error("Python did not report an extension suffix")

    source = ROOT / "src/ehl_solver/_hip_native.cpp"
    output = ROOT / "src/ehl_solver" / f"_hip_native{extension_suffix}"
    command = [
        hipcc,
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        f"--offload-arch={args.arch}",
        f"-I{include_dir}",
        f"-I{sdk / 'include'}",
        str(source),
        f"-L{sdk / 'lib'}",
        f"-Wl,-rpath,{sdk / 'lib'}",
        "-lhipfft",
        "-o",
        str(output),
    ]
    print("Building native HIP extension:")
    print(shlex.join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Built {output}")


if __name__ == "__main__":
    main()
