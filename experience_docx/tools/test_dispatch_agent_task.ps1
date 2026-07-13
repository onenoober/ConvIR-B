param(
    [Parameter(Mandatory = $true)]
    [string]$DispatcherPath,

    [Parameter(Mandatory = $true)]
    [string]$RepositoryLinuxPath,

    [string]$WslDistribution = "Ubuntu-22.04"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-GitValue {
    param([string[]]$Arguments)

    $output = & wsl.exe -d $WslDistribution -- git -C $RepositoryLinuxPath @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "git failed: $($Arguments -join ' ')"
    }
    return ($output | Select-Object -Last 1).Trim()
}

function Copy-Request {
    param($Request)

    return (($Request | ConvertTo-Json -Depth 8) | ConvertFrom-Json)
}

function Invoke-Case {
    param(
        [string]$Name,
        $Request,
        [bool]$ShouldPass,
        [string]$ExpectedModel
    )

    $requestPath = Join-Path $testRoot "$Name.json"
    $Request | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -LiteralPath $requestPath
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $DispatcherPath `
        -RequestPath $requestPath -WslDistribution $WslDistribution 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    $rendered = $output -join "`n"

    if ($ShouldPass) {
        if ($exitCode -ne 0 -or -not $rendered.Contains("MODEL_DISPATCH_DRY_RUN_OK")) {
            throw "$Name expected pass but failed: $rendered"
        }
        if (-not $rendered.Contains('"selected_model"') -or -not $rendered.Contains($ExpectedModel)) {
            throw "$Name did not select $ExpectedModel"
        }
    }
    elseif ($exitCode -eq 0) {
        throw "$Name expected fail but passed"
    }

    return [ordered]@{
        case = $Name
        expected = if ($ShouldPass) { "PASS" } else { "FAIL_CLOSED" }
        observed_exit_code = $exitCode
        expected_model = if ($ShouldPass) { $ExpectedModel } else { $null }
        decision = "PASS"
    }
}

$resolvedDispatcher = (Resolve-Path -LiteralPath $DispatcherPath).Path
$rulesCommit = Invoke-GitValue -Arguments @("rev-parse", "github/main")
$headCommit = Invoke-GitValue -Arguments @("rev-parse", "HEAD")
$testRoot = Join-Path $env:TEMP ("agent-model-dispatcher-tests-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $testRoot | Out-Null

$base = [ordered]@{
    schema_version = 1
    rules_commit = $rulesCommit
    repository_linux_path = $RepositoryLinuxPath
    task_class = "R0_READ_ONLY"
    source_role = "frontier"
    required_role = "fast"
    dispatch_reason = "standalone_repetition"
    effort = "low"
    route_branch_commit = "none"
    route_id = "dispatcher-dry-run-test"
    stage_state = "NOT_APPLICABLE"
    decision = "REPORT_ONLY"
    authorizes = "NONE"
    cloud_status = "inactive"
    next_action = "Return a bounded status field."
    authorization_check = [ordered]@{
        verified = $false
        mechanism = "not_applicable"
        checked_fields = @()
    }
}

$r0 = Copy-Request $base

$r1 = Copy-Request $base
$r1.task_class = "R1_BOUNDED_EXECUTION"
$r1.source_role = "balanced"
$r1.required_role = "fast"
$r1.dispatch_reason = "batch_bounded_operations"
$r1.effort = "medium"
$r1.route_branch_commit = $headCommit
$r1.stage_state = "PASS"
$r1.decision = "CONTINUE"
$r1.authorizes = "RUN_TRACKED_STAGE_ONLY"
$r1.authorization_check.verified = $true
$r1.authorization_check.mechanism = "runner_exact_tuple"
$r1.authorization_check.checked_fields = @("route_id", "state", "decision", "authorizes")

$r2 = Copy-Request $base
$r2.task_class = "R2_ENGINEERING_CONTROL"
$r2.source_role = "frontier"
$r2.required_role = "balanced"
$r2.dispatch_reason = "batch_bounded_operations"
$r2.effort = "medium"
$r2.route_branch_commit = $headCommit
$r2.authorizes = "ENGINEERING_CONTROL_ONLY"

$r3 = Copy-Request $base
$r3.task_class = "R3_SCIENTIFIC_AUTHORITY"
$r3.source_role = "fast"
$r3.required_role = "frontier"
$r3.dispatch_reason = "required_escalation"
$r3.effort = "high"
$r3.route_branch_commit = $headCommit
$r3.authorizes = "SCIENTIFIC_REVIEW_ONLY"

$stale = Copy-Request $r0
$stale.rules_commit = "0000000000000000000000000000000000000000"

$incompleteR1 = Copy-Request $r1
$incompleteR1.authorization_check.checked_fields = @("state")

$underRoleR2 = Copy-Request $r2
$underRoleR2.required_role = "fast"

$sameRole = Copy-Request $r0
$sameRole.source_role = "fast"

$results = @()
$results += Invoke-Case -Name "r0_luna" -Request $r0 -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "r1_luna" -Request $r1 -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "r2_terra" -Request $r2 -ShouldPass $true -ExpectedModel "gpt-5.6-terra"
$results += Invoke-Case -Name "r3_sol" -Request $r3 -ShouldPass $true -ExpectedModel "gpt-5.6-sol"
$results += Invoke-Case -Name "stale_rules" -Request $stale -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "incomplete_r1" -Request $incompleteR1 -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "under_role_r2" -Request $underRoleR2 -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "same_role_short_task" -Request $sameRole -ShouldPass $false -ExpectedModel ""

[ordered]@{
    status = "PASS"
    rules_commit = $rulesCommit
    repository_head = $headCommit
    model_calls = 0
    cases = $results
} | ConvertTo-Json -Depth 8
Write-Output "DISPATCHER_DRY_RUN_TESTS_OK"
