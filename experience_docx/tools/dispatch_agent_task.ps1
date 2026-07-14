param(
    [Parameter(Mandatory = $true)]
    [string]$RequestPath,

    [switch]$Execute,

    [string]$OutputRoot,

    [string]$WslDistribution = "Ubuntu-22.04"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$requiredRequestFields = @(
    "schema_version",
    "rules_commit",
    "repository_linux_path",
    "task_class",
    "source_identity",
    "source_role",
    "source_effort",
    "required_role",
    "dispatch_reason",
    "routing_basis",
    "routing_basis_ref",
    "effort",
    "execution_scope",
    "transport_contract",
    "completion_marker",
    "route_branch_commit",
    "route_id",
    "stage_state",
    "decision",
    "authorizes",
    "cloud_status",
    "next_action",
    "authorization_check"
)
$requiredAuthorizationFields = @("verified", "mechanism", "checked_fields")
$validClasses = @(
    "R0_READ_ONLY",
    "R1_BOUNDED_EXECUTION",
    "R2_ENGINEERING_CONTROL",
    "R3_SCIENTIFIC_AUTHORITY"
)
$validRoles = @("fast", "balanced", "frontier")
$validSourceIdentities = @("unknown", "user_pinned_task", "product_metadata", "cli_status", "dispatcher_receipt")
$validSourceEfforts = @("unknown", "low", "medium", "high", "xhigh")
$validDispatchReasons = @("task_routing", "standalone_repetition", "batch_bounded_operations", "major_handoff")
$validRoutingBases = @("dispatcher_classification", "typed_handoff")
$validExecutionScopes = @("local_read_only", "local_workspace_write", "wsl_workspace_transport", "wsl_cloud_transport")
$validTransportContracts = @("local_only", "tracked_convir_cloud")
$roleRank = @{fast = 0; balanced = 1; frontier = 2}
$minimumRole = @{
    R0_READ_ONLY = "fast"
    R1_BOUNDED_EXECUTION = "fast"
    R2_ENGINEERING_CONTROL = "balanced"
    R3_SCIENTIFIC_AUTHORITY = "frontier"
}
$nonToolItemTypes = @("agent_message", "reasoning", "error")

function Assert-ExactProperties {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string[]]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $actual = @($Object.PSObject.Properties.Name | Sort-Object)
    $wanted = @($Expected | Sort-Object)
    if (($actual -join "`n") -ne ($wanted -join "`n")) {
        throw "$Label fields must be exactly: $($wanted -join ', ')"
    }
}

function Invoke-WslGit {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $stderrPath = Join-Path $env:TEMP ("codex-wsl-git-" + [guid]::NewGuid().ToString("N") + ".stderr")
    $stderrLines = @()
    $output = @()
    $exitCode = $null
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & wsl.exe -d $WslDistribution -- git -C $Repository @Arguments 2> $stderrPath
        $exitCode = $LASTEXITCODE
        if (Test-Path -LiteralPath $stderrPath) {
            $stderrLines = @(Get-Content -LiteralPath $stderrPath)
        }
    }
    finally {
        $ErrorActionPreference = $savedPreference
        if (Test-Path -LiteralPath $stderrPath) {
            Remove-Item -LiteralPath $stderrPath -Force
        }
    }
    if ($exitCode -ne 0) {
        $detail = @($output) + $stderrLines
        throw "git command failed ($exitCode): git -C $Repository $($Arguments -join ' ')`n$($detail -join "`n")"
    }
    return @($output | ForEach-Object { $_.ToString() })
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Text)

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return (($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    finally {
        $sha.Dispose()
    }
}

function Convert-DisplayModelToId {
    param([Parameter(Mandatory = $true)][string]$DisplayName)

    return (($DisplayName.Trim().ToLowerInvariant() -replace "\s+", "-") -replace "[^a-z0-9.-]", "")
}

function Get-TaskWindowsPath {
    param([Parameter(Mandatory = $true)][string]$LinuxPath)

    return "\\wsl.localhost\$WslDistribution" + ($LinuxPath -replace "/", "\")
}

function Read-JsonLines {
    param([Parameter(Mandatory = $true)][string]$Path)

    $events = @()
    foreach ($line in Get-Content -LiteralPath $Path) {
        if (-not [string]::IsNullOrWhiteSpace($line)) {
            $events += ($line | ConvertFrom-Json)
        }
    }
    return $events
}

$totalStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$resolvedRequestPath = (Resolve-Path -LiteralPath $RequestPath).Path
$request = Get-Content -Raw -LiteralPath $resolvedRequestPath | ConvertFrom-Json
Assert-ExactProperties -Object $request -Expected $requiredRequestFields -Label "request"
Assert-ExactProperties -Object $request.authorization_check -Expected $requiredAuthorizationFields -Label "authorization_check"

if ($request.schema_version -ne 2) {
    throw "Unsupported schema_version: $($request.schema_version)"
}
if ($request.rules_commit -notmatch "^[0-9a-f]{40}$") {
    throw "rules_commit must be a full lowercase SHA"
}
if ($request.repository_linux_path -notmatch "^/") {
    throw "repository_linux_path must be absolute"
}
if ($request.task_class -notin $validClasses) {
    throw "Unknown task_class: $($request.task_class)"
}
if ($request.source_identity -notin $validSourceIdentities) {
    throw "Unknown source_identity: $($request.source_identity)"
}
if ($request.required_role -notin $validRoles) {
    throw "Unknown required_role: $($request.required_role)"
}
if ($request.source_role -notin (@("unknown") + $validRoles)) {
    throw "Unknown source_role: $($request.source_role)"
}
if ($request.source_effort -notin $validSourceEfforts) {
    throw "Unknown source_effort: $($request.source_effort)"
}
if ($request.dispatch_reason -notin $validDispatchReasons) {
    throw "Unknown dispatch_reason: $($request.dispatch_reason)"
}
if ($request.routing_basis -notin $validRoutingBases) {
    throw "Unknown routing_basis: $($request.routing_basis)"
}
if ($request.routing_basis_ref -isnot [string] -or [string]::IsNullOrWhiteSpace($request.routing_basis_ref) -or $request.routing_basis_ref.Length -gt 1000) {
    throw "routing_basis_ref must contain 1-1000 characters"
}
if ($request.effort -notin @("low", "medium", "high")) {
    throw "Unknown effort: $($request.effort)"
}
if ($request.execution_scope -notin $validExecutionScopes) {
    throw "Unknown execution_scope: $($request.execution_scope)"
}
if ($request.transport_contract -notin $validTransportContracts) {
    throw "Unknown transport_contract: $($request.transport_contract)"
}
if ($request.completion_marker -notmatch "^[A-Z][A-Z0-9_]{2,127}$") {
    throw "completion_marker must be an uppercase machine marker"
}
if ([string]::IsNullOrWhiteSpace($request.next_action) -or $request.next_action.Length -gt 2000) {
    throw "next_action must contain 1-2000 characters"
}
$nonemptyStringFields = @("route_id", "stage_state", "decision", "authorizes", "cloud_status")
foreach ($field in $nonemptyStringFields) {
    if ($request.$field -isnot [string] -or [string]::IsNullOrWhiteSpace($request.$field)) {
        throw "$field must be a nonempty string"
    }
}
if ($request.authorization_check.mechanism -notin @("not_applicable", "runner_exact_tuple", "balanced_closeout_audit")) {
    throw "Unknown authorization mechanism: $($request.authorization_check.mechanism)"
}
$checkedFields = @($request.authorization_check.checked_fields)
if (@($checkedFields | Select-Object -Unique).Count -ne $checkedFields.Count) {
    throw "authorization_check.checked_fields must be unique"
}
foreach ($field in $checkedFields) {
    if ($field -notin @("route_id", "state", "decision", "authorizes")) {
        throw "Unknown authorization checked field: $field"
    }
}

$sourceUnknown = $request.source_identity -eq "unknown"
if ($sourceUnknown -ne ($request.source_role -eq "unknown") -or $sourceUnknown -ne ($request.source_effort -eq "unknown")) {
    throw "source_identity, source_role, and source_effort must be unknown together"
}
if ($request.routing_basis -eq "typed_handoff" -and $request.routing_basis_ref -eq "none") {
    throw "typed_handoff requires a durable routing_basis_ref"
}
if ($request.routing_basis -eq "typed_handoff" -and $request.routing_basis_ref -notmatch "^github:[0-9a-f]{40}:[A-Za-z0-9._/-]+$") {
    throw "typed_handoff routing_basis_ref must be github:<commit>:<path>"
}
if ($request.routing_basis -ne "typed_handoff" -and $request.routing_basis_ref -ne "none") {
    throw "$($request.routing_basis) requires routing_basis_ref=none"
}
if ($request.execution_scope -eq "wsl_cloud_transport" -and $request.transport_contract -ne "tracked_convir_cloud") {
    throw "wsl_cloud_transport requires transport_contract=tracked_convir_cloud"
}
if ($request.execution_scope -ne "wsl_cloud_transport" -and $request.transport_contract -ne "local_only") {
    throw "non-cloud execution scopes require transport_contract=local_only"
}

$repository = $request.repository_linux_path
$taskWindowsPath = Get-TaskWindowsPath -LinuxPath $repository
if (-not (Test-Path -LiteralPath $taskWindowsPath -PathType Container)) {
    throw "Repository is not visible at $taskWindowsPath"
}

Invoke-WslGit -Repository $repository -Arguments @("fetch", "github", "main") | Out-Null
$currentRulesCommit = (Invoke-WslGit -Repository $repository -Arguments @("rev-parse", "github/main") | Select-Object -Last 1).Trim()
if ($request.rules_commit -ne $currentRulesCommit) {
    throw "STALE_RULES expected=$currentRulesCommit request=$($request.rules_commit)"
}

if ($request.routing_basis -eq "typed_handoff") {
    $basisMatch = [regex]::Match($request.routing_basis_ref, '^github:(?<commit>[0-9a-f]{40}):(?<path>[A-Za-z0-9._/-]+)$')
    $basisCommit = $basisMatch.Groups["commit"].Value
    $basisPath = $basisMatch.Groups["path"].Value
    if ($basisPath.Split('/') -contains "..") {
        throw "typed_handoff path must not contain parent traversal"
    }
    Invoke-WslGit -Repository $repository -Arguments @("fetch", "--quiet", "github", $basisCommit) | Out-Null
    Invoke-WslGit -Repository $repository -Arguments @("cat-file", "-e", "${basisCommit}:$basisPath") | Out-Null
}

$policyLines = Invoke-WslGit -Repository $repository -Arguments @(
    "show",
    "github/main:experience_docx/MODEL_AGENT_COST_ROUTING_PROTOCOL.md"
)
$policy = $policyLines -join "`n"

$mappingPattern = '(?m)^\| `(?<role>frontier|balanced|fast)` \| (?<model>GPT-[^|]+?) \| (?<effort>[^|]+?) \|'
$qualificationPattern = '(?m)^\| `(?<role>frontier|balanced|fast)` / (?<model>GPT-[^|]+?) \| [^|]+ \| `R(?<max>[0-3])` \|'
$mappingMatches = [regex]::Matches($policy, $mappingPattern)
$qualificationMatches = [regex]::Matches($policy, $qualificationPattern)
if ($mappingMatches.Count -ne 3 -or $qualificationMatches.Count -ne 3) {
    throw "Could not parse the canonical role and qualification tables"
}

$modelsByRole = @{}
foreach ($match in $mappingMatches) {
    $modelsByRole[$match.Groups["role"].Value] = Convert-DisplayModelToId $match.Groups["model"].Value
}
$maxClassByRole = @{}
foreach ($match in $qualificationMatches) {
    $maxClassByRole[$match.Groups["role"].Value] = [int]$match.Groups["max"].Value
}

$classLevel = [int]$request.task_class.Substring(1, 1)
$minimum = $minimumRole[$request.task_class]
if ($roleRank[$request.required_role] -lt $roleRank[$minimum]) {
    throw "ROLE_BELOW_MINIMUM class=$($request.task_class) minimum=$minimum request=$($request.required_role)"
}
if ($maxClassByRole[$request.required_role] -lt $classLevel) {
    throw "MODEL_NOT_QUALIFIED role=$($request.required_role) maximum=R$($maxClassByRole[$request.required_role]) request=R$classLevel"
}

$expectedEffort = "medium"
if ($request.required_role -eq "frontier") {
    $expectedEffort = "high"
}
elseif ($request.required_role -eq "fast" -and $classLevel -eq 0) {
    $expectedEffort = "low"
}
if ($request.effort -ne $expectedEffort) {
    throw "EFFORT_MISMATCH expected=$expectedEffort request=$($request.effort)"
}

if ($classLevel -eq 0) {
    if ($request.authorization_check.mechanism -ne "not_applicable" -or $request.authorization_check.verified) {
        throw "R0 authorization_check must be unverified and not_applicable"
    }
}
elseif ($classLevel -eq 1) {
    $requiredChecks = @("authorizes", "decision", "route_id", "state")
    $actualChecks = @($request.authorization_check.checked_fields | Sort-Object)
    if (-not $request.authorization_check.verified -or ($actualChecks -join "`n") -ne ($requiredChecks -join "`n")) {
        throw "R1 requires verified route_id/state/decision/authorizes checks"
    }
    if ($request.authorizes.Trim().ToUpperInvariant() -eq "NONE") {
        throw "R1 authorization tuple authorizes no action"
    }
    if ($request.required_role -eq "fast" -and $request.authorization_check.mechanism -ne "runner_exact_tuple") {
        throw "Fast R1 requires runner_exact_tuple authorization"
    }
    if ($request.required_role -eq "balanced" -and $request.authorization_check.mechanism -notin @("runner_exact_tuple", "balanced_closeout_audit")) {
        throw "Balanced R1 requires an exact runner check or balanced closeout audit"
    }
}

if ($request.route_branch_commit -ne "none") {
    if ($request.route_branch_commit -notmatch "^[0-9a-f]{40}$") {
        throw "route_branch_commit must be none or a full lowercase SHA"
    }
    $repositoryHead = (Invoke-WslGit -Repository $repository -Arguments @("rev-parse", "HEAD") | Select-Object -Last 1).Trim()
    if ($repositoryHead -ne $request.route_branch_commit) {
        throw "ROUTE_COMMIT_MISMATCH repository=$repositoryHead request=$($request.route_branch_commit)"
    }
}
elseif ($classLevel -gt 0) {
    throw "R1-R3 requests require route_branch_commit"
}

$selectedModel = $modelsByRole[$request.required_role]
$sandboxByExecutionScope = @{
    local_read_only = "read-only"
    local_workspace_write = "workspace-write"
    wsl_workspace_transport = "danger-full-access"
    wsl_cloud_transport = "danger-full-access"
}
$sandbox = $sandboxByExecutionScope[$request.execution_scope]
$handoff = [ordered]@{
    schema_version = $request.schema_version
    rules_commit = $currentRulesCommit
    route_branch_commit = $request.route_branch_commit
    route_id = $request.route_id
    task_class = $request.task_class
    source_identity = $request.source_identity
    source_role = $request.source_role
    source_effort = $request.source_effort
    required_role = $request.required_role
    dispatch_reason = $request.dispatch_reason
    routing_basis = $request.routing_basis
    routing_basis_ref = $request.routing_basis_ref
    effort = $request.effort
    execution_scope = $request.execution_scope
    transport_contract = $request.transport_contract
    completion_marker = $request.completion_marker
    stage_state = $request.stage_state
    decision = $request.decision
    authorizes = $request.authorizes
    cloud_status = $request.cloud_status
    next_action = $request.next_action
}
$handoffJson = $handoff | ConvertTo-Json -Compress
$handoffSha = Get-Sha256 -Text $handoffJson
$routeMarker = "MODEL_ROUTE class=$($request.task_class) role=$($request.required_role) effort=$($request.effort)"
$handoffAck = "HANDOFF_ACK sha256=$handoffSha"
$basisInstruction = if ($request.routing_basis -eq "typed_handoff") {
    "Before the next action, read and verify routing_basis_ref=$($request.routing_basis_ref); fail closed if it does not match the handoff JSON."
}
else {
    "The host acted only as a dispatcher and applied the canonical task-class table; if the class is ambiguous, stop before the next action and request R3/frontier routing."
}
$promptLines = @(
    'Use $experiment-model-router for this task.',
    "The deterministic external dispatcher selected model=$selectedModel from github/main@$currentRulesCommit.",
    "Before any tool call, send a progress message containing exactly these two lines:",
    $routeMarker,
    $handoffAck,
    "Do not downgrade the role or expand the task scope.",
    $(if ($request.execution_scope -eq "wsl_workspace_transport") { "This scope permits WSL local workspace transport only. Do not call convir_remote_script.sh, convir_route_ tools, convir-ops, SSH, or cloud commands." }),
    $(if ($request.transport_contract -eq "tracked_convir_cloud") { "Cloud access is limited to the tracked ConvIR transport named by the bounded next action; do not construct arbitrary SSH commands." }),
    $basisInstruction,
    "If new evidence requires a stronger role, stop before the next write or decision and emit MODEL_SWITCH_REQUIRED with a new dispatcher request.",
    "Handoff JSON: $handoffJson",
    "Perform exactly this next action: $($request.next_action)",
    "Report success only after the action's success conditions are verified. On success, end the final answer with this exact marker on its own line: $($request.completion_marker)"
)
$childPrompt = $promptLines -join [Environment]::NewLine
$prelaunchSeconds = [Math]::Round($totalStopwatch.Elapsed.TotalSeconds, 3)

$dispatchPlan = [ordered]@{
    dispatcher_status = "DRY_RUN_OK"
    schema_version = $request.schema_version
    rules_commit = $currentRulesCommit
    task_class = $request.task_class
    source_identity = $request.source_identity
    source_role = $request.source_role
    source_effort = $request.source_effort
    selected_role = $request.required_role
    dispatch_reason = $request.dispatch_reason
    routing_basis = $request.routing_basis
    routing_basis_ref = $request.routing_basis_ref
    selected_model = $selectedModel
    effort = $request.effort
    execution_scope = $request.execution_scope
    transport_contract = $request.transport_contract
    completion_marker = $request.completion_marker
    sandbox = $sandbox
    repository_linux_path = $repository
    handoff_sha256 = $handoffSha
    prelaunch_seconds = $prelaunchSeconds
    model_calls_before_launch = 0
    output_root = $null
}

if (-not $Execute) {
    $dispatchPlan | ConvertTo-Json -Depth 6
    Write-Output "MODEL_DISPATCH_DRY_RUN_OK"
    exit 0
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $safeRoute = $request.route_id -replace "[^A-Za-z0-9._-]", "_"
    $runName = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ") + "-$safeRoute-$($request.required_role)"
    $OutputRoot = Join-Path (Join-Path $env:USERPROFILE ".codex\dispatcher-runs") $runName
}
$resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $resolvedOutputRoot) {
    throw "Refusing to overwrite output root: $resolvedOutputRoot"
}
New-Item -ItemType Directory -Path $resolvedOutputRoot | Out-Null
$promptPath = Join-Path $resolvedOutputRoot "child_prompt.txt"
$eventsPath = Join-Path $resolvedOutputRoot "events.jsonl"
$stderrPath = Join-Path $resolvedOutputRoot "stderr.log"
$answerPath = Join-Path $resolvedOutputRoot "answer.txt"
$metadataPath = Join-Path $resolvedOutputRoot "dispatch_metadata.json"
$childPrompt | Set-Content -Encoding utf8 -LiteralPath $promptPath

$codex = Get-Command codex.cmd -ErrorAction Stop
$savedPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$cliVersion = (& $codex.Source --version 2>$null).Trim()
$ErrorActionPreference = $savedPreference
$arguments = @(
    "exec",
    "--model", $selectedModel,
    "-c", ('model_reasoning_effort="' + $request.effort + '"'),
    "--sandbox", $sandbox,
    "--ephemeral",
    "--output-last-message", $answerPath,
    "--json",
    "--cd", $taskWindowsPath,
    "-"
)

$launchStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$ErrorActionPreference = "Continue"
$childExitCode = $null
Push-Location -LiteralPath $env:TEMP
try {
    $childPrompt | & $codex.Source @arguments 1> $eventsPath 2> $stderrPath
    $childExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    $ErrorActionPreference = $savedPreference
}
$launchStopwatch.Stop()
$totalStopwatch.Stop()

$events = Read-JsonLines -Path $eventsPath
$ackIndex = -1
$firstToolIndex = -1
$turnCompleted = $false
$switchRequired = $false
$usage = $null
for ($index = 0; $index -lt $events.Count; $index++) {
    $event = $events[$index]
    if ($event.type -eq "turn.completed") {
        $turnCompleted = $true
        $usage = $event.usage
    }
    if ($event.type -in @("item.started", "item.completed") -and $null -ne $event.item) {
        if ($event.item.type -eq "agent_message") {
            $text = [string]$event.item.text
            if ($ackIndex -lt 0 -and $text.Contains($routeMarker) -and $text.Contains($handoffAck)) {
                $ackIndex = $index
            }
            if ($text.Contains("MODEL_SWITCH_REQUIRED")) {
                $switchRequired = $true
            }
        }
        if ($event.type -eq "item.started" -and $event.item.type -notin $nonToolItemTypes -and $firstToolIndex -lt 0) {
            $firstToolIndex = $index
        }
    }
}
$toolBeforeAck = $firstToolIndex -ge 0 -and ($ackIndex -lt 0 -or $firstToolIndex -lt $ackIndex)
$answerLines = @()
if (Test-Path -LiteralPath $answerPath) {
    $answerLines = @(Get-Content -LiteralPath $answerPath | ForEach-Object { $_.Trim() })
}
$completionMarkerSeen = $answerLines -contains $request.completion_marker
$dispatchPassed = $childExitCode -eq 0 -and $turnCompleted -and $ackIndex -ge 0 -and -not $toolBeforeAck -and -not $switchRequired -and $completionMarkerSeen
$status = if ($dispatchPassed) { "PASS" } else { "FAIL" }

$metadata = [ordered]@{
    dispatcher_status = $status
    schema_version = $request.schema_version
    rules_commit = $currentRulesCommit
    cli_version = $cliVersion
    task_class = $request.task_class
    source_identity = $request.source_identity
    source_role = $request.source_role
    source_effort = $request.source_effort
    selected_role = $request.required_role
    dispatch_reason = $request.dispatch_reason
    routing_basis = $request.routing_basis
    routing_basis_ref = $request.routing_basis_ref
    selected_model = $selectedModel
    effort = $request.effort
    execution_scope = $request.execution_scope
    transport_contract = $request.transport_contract
    completion_marker = $request.completion_marker
    sandbox = $sandbox
    repository_linux_path = $repository
    handoff_sha256 = $handoffSha
    route_marker = $routeMarker
    child_exit_code = $childExitCode
    turn_completed = $turnCompleted
    handoff_ack_seen = $ackIndex -ge 0
    tool_before_ack = $toolBeforeAck
    switch_required = $switchRequired
    completion_marker_seen = $completionMarkerSeen
    prelaunch_seconds = $prelaunchSeconds
    child_elapsed_seconds = [Math]::Round($launchStopwatch.Elapsed.TotalSeconds, 3)
    total_elapsed_seconds = [Math]::Round($totalStopwatch.Elapsed.TotalSeconds, 3)
    model_calls_before_launch = 0
    usage = $usage
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -LiteralPath $metadataPath

if (-not $dispatchPassed) {
    throw "MODEL_DISPATCH_FAILED output=$resolvedOutputRoot"
}
Write-Output "MODEL_DISPATCH_OK model=$selectedModel role=$($request.required_role) output=$resolvedOutputRoot"
