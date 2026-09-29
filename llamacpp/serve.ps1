<#
.SYNOPSIS
    Start llama-server (OpenAI-compatible API) on a GGUF model.
.EXAMPLE
    .\llamacpp\serve.ps1                                   # F16 baseline on port 8080
    .\llamacpp\serve.ps1 -Model models\gguf\x-q4_k_m.gguf -ExtraArgs "--parallel","4"
#>
param(
    [string]$Model,
    [int]$Port = 8080,
    [int]$CtxSize = 4096,
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"

if (-not $Model) { $Model = $F16Gguf }
if (-not (Test-Path $Model)) { throw "Model not found: $Model (run setup.ps1 and convert.ps1 first)" }

$server = Join-Path $LlamaBinDir "llama-server.exe"
# --alias sets the model name the API reports and accepts, independent of the file name.
& $server --model $Model --alias $ModelSlug --port $Port --ctx-size $CtxSize @ExtraArgs
