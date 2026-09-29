<#
.SYNOPSIS
    Quantize the F16 GGUF into each level used by the study. Existing outputs are skipped.
#>
param(
    [string[]]$Types = @("Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"

if (-not (Test-Path $F16Gguf)) { throw "F16 GGUF not found: $F16Gguf (run convert.ps1 first)" }
$quantize = Join-Path $LlamaBinDir "llama-quantize.exe"

foreach ($type in $Types) {
    $out = Join-Path $GgufDir ("$ModelSlug-{0}.gguf" -f $type.ToLower())
    if (Test-Path $out) {
        Write-Host "Exists: $out"
        continue
    }
    Write-Host "Quantizing -> $type"
    & $quantize $F16Gguf $out $type
    if ($LASTEXITCODE -ne 0) { throw "llama-quantize failed for $type" }
    Write-Host ("  {0:N2} GB" -f ((Get-Item $out).Length / 1GB))
}
