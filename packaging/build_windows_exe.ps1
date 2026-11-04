$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$EntryPoint = Join-Path $Root 'src\ehl_solver\prototype_cli.py'
$ExampleCase = Join-Path $Root 'examples\example_case_50n_33.json'
$Dist = Join-Path $Root 'dist\windows'
$Build = Join-Path ([System.IO.Path]::GetTempPath()) "ehl-pyinstaller-$([guid]::NewGuid())"
$Executable = Join-Path $Dist 'ehl-run.exe'

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )

    & $FilePath @Arguments
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "$Description failed with exit code $ExitCode."
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python virtual environment not found at $Python. Create it with: python -m venv .venv"
}
foreach ($RequiredFile in @($EntryPoint, $ExampleCase)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required packaging input not found: $RequiredFile"
    }
}

Set-Location $Root
& $Python -m pip show PyInstaller *> $null
if ($LASTEXITCODE -ne 0) {
    Invoke-CheckedNative $Python @('-m', 'pip', 'install', 'PyInstaller') 'PyInstaller installation'
}

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
try {
    Invoke-CheckedNative $Python @(
        '-m', 'PyInstaller',
        '--onefile', '--console', '--noconfirm', '--clean',
        '--exclude-module', 'torch',
        '--name', 'ehl-run',
        '--paths', (Join-Path $Root 'src'),
        '--distpath', $Dist,
        '--workpath', $Build,
        '--specpath', $Build,
        $EntryPoint
    ) 'PyInstaller build'
}
finally {
    if (Test-Path -LiteralPath $Build -PathType Container) {
        Remove-Item -LiteralPath $Build -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "PyInstaller returned success but the executable was not created: $Executable"
}
if ((Get-Item -LiteralPath $Executable).Length -eq 0) {
    throw "PyInstaller created an empty executable: $Executable"
}

Invoke-CheckedNative $Executable @('--help') 'Executable --help smoke test'

$SmokeOutput = Join-Path ([System.IO.Path]::GetTempPath()) "ehl-run-smoke-$([guid]::NewGuid())"
try {
    Invoke-CheckedNative $Executable @(
        '--case', $ExampleCase,
        '--output', $SmokeOutput
    ) 'Packaged example-case smoke test'

    foreach ($Name in @('report.md', 'fields.png', 'summary.json', 'solution.npz')) {
        if (-not (Test-Path -LiteralPath (Join-Path $SmokeOutput $Name) -PathType Leaf)) {
            throw "Packaged smoke test did not produce expected output: $Name"
        }
    }
    $Summary = Get-Content -LiteralPath (Join-Path $SmokeOutput 'summary.json') -Raw |
        ConvertFrom-Json
    if ($Summary.status -ne 'accepted_numerical_root') {
        throw "Packaged example case was not accepted: $($Summary.status)"
    }
}
finally {
    if (Test-Path -LiteralPath $SmokeOutput -PathType Container) {
        Remove-Item -LiteralPath $SmokeOutput -Recurse -Force
    }
}

Write-Host "Built and smoke-tested: $Executable"
