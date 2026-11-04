# Windows packaging and usage

This folder contains a PyInstaller builder for a Windows `.exe` wrapper around
the prototype CLI. The builder uses the repository `.venv`, installs
PyInstaller there only if it is missing, and fails immediately when a native
build or smoke-test command returns a nonzero exit code. The executable is
CPU-only and excludes optional PyTorch to avoid bundling its large GPU/runtime
payload.

## Build the EXE

From a PowerShell prompt in the repo root:

```powershell
.\packaging\build_windows_exe.ps1
```

The script creates a single-file executable at:

```text
dist\windows\ehl-run.exe
```

The build script verifies that the file exists, checks `--help`, runs the
bundled example case in a temporary directory, and checks its numerical status
and output files. This is a smoke test in the build environment, not evidence
that the executable runs on a clean Windows machine.

## Run the packaged app

```powershell
.\dist\windows\ehl-run.exe --case .\examples\example_case_50n_33.json --output .\results\example_packaged_run
```

This launches the same solver path as the Python entry point and writes the usual output files:

- `report.md`
- `fields.png`
- `summary.json`
- `solution.npz`

## Example case

The sample case is stored in:

```text
examples\example_case_50n_33.json
```

It is a 50 N circular point-contact case that is small enough for quick local validation and demonstration runs.

The executable packages the research prototype; it does not establish physical
validation or engineering accuracy. Before distributing it, test it on a clean
supported Windows installation and confirm any platform runtime requirements.
