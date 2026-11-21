param(
    [Parameter(Mandatory=$true)][string]$SpeechCheckout,
    [Parameter(Mandatory=$true)][string]$SpeechBuild,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [Parameter(Mandatory=$true)][string]$CudaToolkit
)
$ErrorActionPreference = 'Stop'
$speechSource = (Resolve-Path -LiteralPath $SpeechCheckout).Path
$speechBinary = (Resolve-Path -LiteralPath $SpeechBuild).Path
$cudaRoot = (Resolve-Path -LiteralPath $CudaToolkit).Path
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Run from an initialized Visual Studio x64 developer environment.'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$probeOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path
$probeObject = Join-Path $probeOutput 'speech_direct_probe.obj'
$probeExecutable = Join-Path $probeOutput 'speech_direct_probe.exe'
if ((Test-Path -LiteralPath $probeObject) -or (Test-Path -LiteralPath $probeExecutable)) {
    throw 'Use a fresh output directory; retained probes are not overwritten.'
}
$probeSource = Join-Path $PSScriptRoot 'speech_direct_probe.cpp'
$includeArgs = @('/I' + (Join-Path $speechSource 'src/common'),
                 '/I' + (Join-Path $speechSource 'src/families'),
                 '/I' + (Join-Path $speechSource 'ggml/include'))
& cl.exe /nologo /std:c++17 /MD /O2 /DNDEBUG /utf-8 /EHsc /c @includeArgs "/Fo$probeObject" $probeSource
if ($LASTEXITCODE -ne 0) { throw 'Probe compilation failed.' }
$libraries = @('qwen3tts.lib', 'speechcommon.lib', 'ggml/src/ggml.lib',
               'ggml/src/ggml-cpu.lib', 'ggml/src/ggml-cuda/ggml-cuda.lib', 'ggml/src/ggml-base.lib') |
    ForEach-Object { Join-Path $speechBinary $_ }
$libraries += @('cudart.lib', 'cublas.lib', 'cublasLt.lib', 'cuda.lib') |
    ForEach-Object { Join-Path $cudaRoot ('lib/x64/' + $_) }
& link.exe /NOLOGO "/OUT:$probeExecutable" $probeObject @libraries shell32.lib
if ($LASTEXITCODE -ne 0) { throw 'Probe linking failed.' }
