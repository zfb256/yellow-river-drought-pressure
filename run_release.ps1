# Full pipeline in one Python environment (requirements.txt); -From N resumes at step N.
param([string]$Python = 'python', [int]$From = 0)
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$steps = @(
  @('aggregate_products.py', 'cmfd'), @('aggregate_products.py', 'gleam'), @('aggregate_basins.py'),
  @('aggregate_grun.py'), @('drought_indices.py'), @('check_homogeneity.py'), @('pet_check.py'),
  @('drought_indices_v2.py'), @('validation.py'), @('nested_basins.py'), @('mechanism_analysis.py'),
  @('mechanism_robustness.py'), @('mechanism_undammed.py'), @('counterfactual.py'),
  @('counterfactual.py', '--snow-shift=-1'), @('counterfactual.py', '--snow-shift=1'), @('demand_checks.py'), @('pet_drivers.py'),
  @('study_figures.py'), @('make_mechanism_figures.py'), @('check_mechanisms.py'))
foreach ($s in $steps[$From..($steps.Length - 1)]) {
  $log = "logs/" + (($s -join '_') -replace '\.py', '') + ".log"
  $extra = @(); if ($s.Length -gt 1) { $extra = $s[1..($s.Length - 1)] }
  & $Python "scripts/$($s[0])" $extra 2>&1 | ForEach-Object { "$_" } | Set-Content -Encoding utf8 $log
  if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $($s -join ' ')"; exit 1 }
  Write-Output "done: $($s -join ' ')"
}
Write-Output 'ALL DONE'
