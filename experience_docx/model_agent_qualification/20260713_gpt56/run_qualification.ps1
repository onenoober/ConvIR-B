param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [ValidateSet("low", "medium", "high")]
    [string]$Effort,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [switch]$UseServerSchema
)

$ErrorActionPreference = "Stop"
$sourceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

if (Test-Path -LiteralPath $resolvedOutputRoot) {
    throw "Refusing to overwrite existing output root: $resolvedOutputRoot"
}

$inputRoot = Join-Path $resolvedOutputRoot "input"
New-Item -ItemType Directory -Path $inputRoot | Out-Null

$inputNames = @(
    "qualification_prompt.md",
    "cases.json",
    "response.schema.json"
)
foreach ($name in $inputNames) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $name) -Destination $inputRoot
}

$instructions = Get-Content -Raw -LiteralPath (Join-Path $inputRoot "qualification_prompt.md")
$cases = Get-Content -Raw -LiteralPath (Join-Path $inputRoot "cases.json")
$schema = Get-Content -Raw -LiteralPath (Join-Path $inputRoot "response.schema.json")
$newline = [Environment]::NewLine
$combinedPrompt = $instructions + $newline + $newline + "cases.json:" + $newline + $cases +
    $newline + $newline + "response.schema.json:" + $newline + $schema

$eventsPath = Join-Path $resolvedOutputRoot "events.jsonl"
$stderrPath = Join-Path $resolvedOutputRoot "stderr.log"
$answerPath = Join-Path $resolvedOutputRoot "answer.json"
$schemaPath = Join-Path $inputRoot "response.schema.json"
$codex = Get-Command codex -ErrorAction Stop
$savedErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$cliVersion = (& $codex.Source --version 2>$null).Trim()
$ErrorActionPreference = $savedErrorPreference

$arguments = @(
    "exec",
    "--model", $Model,
    "-c", ('model_reasoning_effort="' + $Effort + '"'),
    "--sandbox", "read-only",
    "--ephemeral",
    "--skip-git-repo-check",
    "--output-last-message", $answerPath,
    "--json",
    "--cd", $inputRoot,
    "-"
)
if ($UseServerSchema) {
    $arguments = $arguments[0..8] + @("--output-schema", $schemaPath) + $arguments[9..($arguments.Count - 1)]
}

$startedAt = [DateTimeOffset]::UtcNow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$ErrorActionPreference = "Continue"
$combinedPrompt | & $codex.Source @arguments 1> $eventsPath 2> $stderrPath
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorPreference
$stopwatch.Stop()
$finishedAt = [DateTimeOffset]::UtcNow

$metadata = [ordered]@{
    model = $Model
    effort = $Effort
    cli_version = $cliVersion
    started_at_utc = $startedAt.ToString("o")
    finished_at_utc = $finishedAt.ToString("o")
    elapsed_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    exit_code = $exitCode
    server_schema = [bool]$UseServerSchema
    cases_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $inputRoot "cases.json")).Hash.ToLowerInvariant()
    prompt_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $inputRoot "qualification_prompt.md")).Hash.ToLowerInvariant()
    schema_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $schemaPath).Hash.ToLowerInvariant()
}
$metadata | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $resolvedOutputRoot "run_metadata.json")

if ($exitCode -ne 0) {
    throw "codex exec failed with exit code $exitCode; see $stderrPath"
}
if (-not (Test-Path -LiteralPath $answerPath)) {
    throw "codex exec did not create $answerPath"
}

Write-Output "QUALIFICATION_RUN_OK model=$Model effort=$Effort output=$resolvedOutputRoot"
