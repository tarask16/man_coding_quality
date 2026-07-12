param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("positive", "null")]
    [string]$Control = "positive",

    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path $ProjectRoot).Path
$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $project "synthetic_runs\$Control"
$python = Join-Path $project ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Интерпретатор виртуального окружения не найден: $python"
}

$dataset = Join-Path `
    $packageRoot `
    "data\${Control}_control\validation_dataset_test.csv"

if (-not (Test-Path $dataset)) {
    throw "Синтетический тестовый датасет не найден: $dataset"
}

Remove-Item $runRoot -Recurse -Force -ErrorAction SilentlyContinue
New-Item $runRoot -ItemType Directory | Out-Null
Copy-Item (Join-Path $project "src") $runRoot -Recurse
Copy-Item (Join-Path $project "configs") $runRoot -Recurse
New-Item (Join-Path $runRoot "reports\chapter6") -ItemType Directory -Force | Out-Null
Copy-Item $dataset (Join-Path $runRoot "reports\chapter6\validation_dataset.csv")
Copy-Item `
    (Join-Path $packageRoot "configs\chapter6_synthetic_test.yaml") `
    (Join-Path $runRoot "configs\chapter6.yaml") `
    -Force

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $runRoot "src"

try {
    Push-Location $runRoot
    & $python -m manual_coding_sim.validation.chapter6_runner `
        --project-root . `
        --config configs\chapter6.yaml `
        --compare-baselines `
        --bootstrap-analysis

    if ($LASTEXITCODE -ne 0) {
        throw "Синтетическая проверка завершилась с кодом $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}

Write-Host "Синтетический контроль выполнен в изолированном каталоге: $runRoot"
Write-Host "Основные отчеты: $runRoot\reports\chapter6"
