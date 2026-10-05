<#
.SYNOPSIS
    Start llama-server (OpenAI-compatible API) with the configuration chosen in Phase 4.
.DESCRIPTION
    Defaults come from the measurements in results/ on an i5-1135G7 (4 cores / 8 threads):
      Q5_K_M     best quality/speed trade-off (results/quant_sweep.json)
      4 threads  generation plateaus at the physical core count (runtime_threads.json)
      4 slots    +49% throughput, ~40x lower median TTFT at 4 concurrent requests
                 vs 1 slot (llamacpp-q5_k_m-parallel{1,4}.json)
      FA on      neutral to slightly positive on this CPU (runtime_attention.json)
      KV f16     q8_0 halves KV memory but cuts prompt speed up to 2.6x at 2k context
      unified KV all slots share one CtxSize-token pool; without it an explicit --parallel
                 splits the context (4096 / 4 slots = 1024 tokens each)
.EXAMPLE
    .\llamacpp\serve.ps1                                         # tuned Q5_K_M setup
    .\llamacpp\serve.ps1 -Quant f16                              # unquantized baseline
    .\llamacpp\serve.ps1 -Quant q4_k_m -Parallel 1 -ExtraArgs "-ctk","q8_0","-ctv","q8_0"
#>
param(
    [string]$Quant = "q5_k_m",
    [string]$Model,
    [int]$Port = 8080,
    [int]$CtxSize = 4096,
    [int]$Threads = 4,
    [int]$Parallel = 4,
    [ValidateSet("on", "off", "auto")]
    [string]$FlashAttn = "on",
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"

if (-not $Model) { $Model = Join-Path $GgufDir "$ModelSlug-$($Quant.ToLower()).gguf" }
if (-not (Test-Path $Model)) {
    throw "Model not found: $Model (run setup.ps1, convert.ps1 and quantize.ps1 first)"
}

$server = Join-Path $LlamaBinDir "llama-server.exe"
# --alias sets the model name the API reports and accepts, independent of the file name.
& $server --model $Model --alias $ModelSlug --port $Port --ctx-size $CtxSize `
    --threads $Threads --parallel $Parallel --kv-unified --flash-attn $FlashAttn @ExtraArgs
