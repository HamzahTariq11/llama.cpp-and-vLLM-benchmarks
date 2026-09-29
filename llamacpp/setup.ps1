<#
.SYNOPSIS
    Download the pinned llama.cpp Windows CPU build, its matching source (for the
    GGUF conversion script), and the Hugging Face model weights. Safe to re-run:
    anything already present is skipped.
#>
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\config.ps1"

New-Item -ItemType Directory -Force $ToolsDir, $ModelsDir | Out-Null

function Get-Zip([string]$Url, [string]$Dest) {
    $zip = "$Dest.zip"
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $Dest -Force
    Remove-Item $zip
}

# 1. Prebuilt binaries (llama-server, llama-quantize, llama-bench, llama-perplexity, ...)
if (-not (Test-Path (Join-Path $LlamaBinDir "llama-server.exe"))) {
    $url = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/llama-$LlamaTag-bin-win-cpu-x64.zip"
    Get-Zip $url $LlamaBinDir
} else {
    Write-Host "llama.cpp binaries present: $LlamaBinDir"
}

# 2. Matching source tree: provides convert_hf_to_gguf.py and the gguf-py package it uses.
if (-not (Test-Path (Join-Path $LlamaSrcDir "convert_hf_to_gguf.py"))) {
    $url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LlamaTag.zip"
    $tmp = "$LlamaSrcDir-tmp"
    Get-Zip $url $tmp
    # The archive contains a single top-level folder (llama.cpp-<tag>); flatten it.
    Move-Item (Get-ChildItem $tmp | Select-Object -First 1).FullName $LlamaSrcDir
    Remove-Item $tmp
} else {
    Write-Host "llama.cpp source present: $LlamaSrcDir"
}

# 3. Hugging Face weights, tokenizer and config (the repo holds nothing else of note).
if (-not (Test-Path (Join-Path $HfModelDir "config.json"))) {
    Write-Host "Downloading $ModelId"
    uv run hf download $ModelId --local-dir $HfModelDir
    if ($LASTEXITCODE -ne 0) { throw "Model download failed" }
} else {
    Write-Host "HF weights present: $HfModelDir"
}

# 4. Perplexity eval text: wikitext-2 test split, the same file llama.cpp's own scripts use.
if (-not (Test-Path $WikitextFile)) {
    Get-Zip "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" (Join-Path $ToolsDir "wikitext")
} else {
    Write-Host "wikitext-2 present: $WikitextFile"
}

Write-Host "Setup complete."
