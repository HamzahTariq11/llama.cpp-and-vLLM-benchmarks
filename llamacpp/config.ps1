# Shared settings for the llama.cpp scripts. Dot-source this file: . "$PSScriptRoot\config.ps1"

# Pinned llama.cpp build, so results are reproducible.
$LlamaTag = "b11254"

# Hugging Face model used throughout the project.
$ModelId = "Qwen/Qwen2.5-1.5B-Instruct"
$ModelSlug = "qwen2.5-1.5b-instruct"

# Paths (all gitignored).
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ToolsDir = Join-Path $RepoRoot "tools"
$LlamaBinDir = Join-Path $ToolsDir "llama.cpp-$LlamaTag-bin"
$LlamaSrcDir = Join-Path $ToolsDir "llama.cpp-$LlamaTag-src"
$ModelsDir = Join-Path $RepoRoot "models"
$HfModelDir = Join-Path $ModelsDir "hf\$ModelSlug"
$GgufDir = Join-Path $ModelsDir "gguf"
$F16Gguf = Join-Path $GgufDir "$ModelSlug-f16.gguf"
$WikitextFile = Join-Path $ToolsDir "wikitext\wikitext-2-raw\wiki.test.raw"
