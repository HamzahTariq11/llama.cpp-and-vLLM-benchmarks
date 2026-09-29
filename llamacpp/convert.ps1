<#
.SYNOPSIS
    Convert the Hugging Face weights to an F16 GGUF using llama.cpp's own converter.
    The converter's Python deps live in the `convert` uv dependency group.
#>
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"

if (Test-Path $F16Gguf) {
    Write-Host "Already converted: $F16Gguf"
    exit 0
}
New-Item -ItemType Directory -Force $GgufDir | Out-Null

$script = Join-Path $LlamaSrcDir "convert_hf_to_gguf.py"
uv run --group convert python $script $HfModelDir --outtype f16 --outfile $F16Gguf
if ($LASTEXITCODE -ne 0) { throw "Conversion failed" }

Write-Host ("Wrote {0} ({1:N2} GB)" -f $F16Gguf, ((Get-Item $F16Gguf).Length / 1GB))
