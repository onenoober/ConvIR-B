param(
    [Parameter(Mandatory = $true)]
    [string]$DispatcherPath,

    [Parameter(Mandatory = $true)]
    [string]$RepositoryLinuxPath,

    [string]$WslDistribution = "Ubuntu-22.04"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$isLinuxHost = $PSVersionTable.PSEdition -eq "Core" -and [System.IO.Path]::DirectorySeparatorChar -eq '/'
$powerShellHost = if ($isLinuxHost) { Join-Path $PSHOME "pwsh" } else { "powershell.exe" }

function Invoke-GitValue {
    param([string[]]$Arguments)

    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($isLinuxHost) {
            $output = & git -C $RepositoryLinuxPath @Arguments 2>$null
        }
        else {
            $output = & wsl.exe -d $WslDistribution -- git -C $RepositoryLinuxPath @Arguments 2>$null
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($exitCode -ne 0) {
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
    $output = & $powerShellHost -NoProfile -ExecutionPolicy Bypass -File $DispatcherPath `
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
    schema_version = 2
    rules_commit = $rulesCommit
    repository_linux_path = $RepositoryLinuxPath
    task_class = "R0_READ_ONLY"
    source_identity = "user_pinned_task"
    source_role = "frontier"
    source_effort = "high"
    required_role = "fast"
    dispatch_reason = "task_routing"
    routing_basis = "dispatcher_classification"
    routing_basis_ref = "none"
    effort = "low"
    execution_scope = "local_read_only"
    transport_contract = "local_only"
    completion_marker = "DISPATCHER_DRY_RUN_TEST_OK"
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
$r1.source_identity = "user_pinned_task"
$r1.source_role = "balanced"
$r1.source_effort = "medium"
$r1.required_role = "fast"
$r1.dispatch_reason = "batch_bounded_operations"
$r1.effort = "medium"
$r1.execution_scope = "wsl_cloud_transport"
$r1.transport_contract = "tracked_convir_cloud"
$r1.next_action = "Use convir_route_plan_manifest, convir_route_start_authorized, and convir_route_finish for one authorized stage package."
$r1.route_branch_commit = $headCommit
$r1.route_id = "haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714"
$r1.stage_state = "COMPLETED_GATE_PASS"
$r1.decision = "V4A_A0R_REPRODUCTION_PASS_AUTHORIZE_A0D_AND_A0P"
$r1.authorizes = "A0D_AND_A0P_ONLY"
$r1.routing_basis = "typed_handoff"
$r1.routing_basis_ref = "github:${rulesCommit}:experience_docx/experiment_logs/haze4k_v5_chd_rm_v4a_conditional_safety_audit_20260714/v4a_a0r_closeout.json"
$r1.authorization_check.verified = $true
$r1.authorization_check.mechanism = "runner_exact_tuple"
$r1.authorization_check.checked_fields = @("route_id", "state", "decision", "authorizes")

$r2 = Copy-Request $base
$r2.task_class = "R2_ENGINEERING_CONTROL"
$r2.source_identity = "user_pinned_task"
$r2.source_role = "frontier"
$r2.source_effort = "high"
$r2.required_role = "balanced"
$r2.dispatch_reason = "batch_bounded_operations"
$r2.effort = "medium"
$r2.execution_scope = "local_workspace_write"
$r2.route_branch_commit = $headCommit
$r2.authorizes = "ENGINEERING_CONTROL_ONLY"

$r3 = Copy-Request $base
$r3.task_class = "R3_SCIENTIFIC_AUTHORITY"
$r3.source_identity = "user_pinned_task"
$r3.source_role = "fast"
$r3.source_effort = "medium"
$r3.required_role = "frontier"
$r3.dispatch_reason = "task_routing"
$r3.effort = "high"
$r3.execution_scope = "local_workspace_write"
$r3.route_branch_commit = $headCommit
$r3.authorizes = "SCIENTIFIC_REVIEW_ONLY"

$stale = Copy-Request $r0
$stale.rules_commit = "0000000000000000000000000000000000000000"

$incompleteR1 = Copy-Request $r1
$incompleteR1.authorization_check.checked_fields = @("state")

$mismatchedR1 = Copy-Request $r1
$mismatchedR1.decision = "TAMPERED_DECISION"

$unboundR1 = Copy-Request $r1
$unboundR1.routing_basis = "dispatcher_classification"
$unboundR1.routing_basis_ref = "none"

$legacyR1 = Copy-Request $r1
$legacyR1.next_action = "Call convir_route_monitor with the existing receipt."

$underRoleR2 = Copy-Request $r2
$underRoleR2.required_role = "fast"

$wslWorkspace = Copy-Request $r2
$wslWorkspace.execution_scope = "wsl_workspace_transport"
$wslWorkspace.next_action = "Use WSL only to repair the local dispatcher fixtures."

$wslWorkspaceCloud = Copy-Request $wslWorkspace
$wslWorkspaceCloud.transport_contract = "tracked_convir_cloud"

$sameRole = Copy-Request $r0
$sameRole.source_role = "fast"
$sameRole.source_effort = "low"

$unknownClassifiedR0 = Copy-Request $r0
$unknownClassifiedR0.source_identity = "unknown"
$unknownClassifiedR0.source_role = "unknown"
$unknownClassifiedR0.source_effort = "unknown"

$unknownTyped = Copy-Request $unknownClassifiedR0
$unknownTyped.routing_basis = "typed_handoff"
$unknownTyped.routing_basis_ref = "github:${rulesCommit}:experience_docx/model_agent_dispatcher/20260713/README.md"

$unknownR1 = Copy-Request $r1
$unknownR1.source_identity = "unknown"
$unknownR1.source_role = "unknown"
$unknownR1.source_effort = "unknown"

$unknownR2 = Copy-Request $r2
$unknownR2.source_identity = "unknown"
$unknownR2.source_role = "unknown"
$unknownR2.source_effort = "unknown"
$unknownR2.dispatch_reason = "task_routing"
$unknownR2.routing_basis = "dispatcher_classification"
$unknownR2.routing_basis_ref = "none"

$unknownR3 = Copy-Request $r3
$unknownR3.source_identity = "unknown"
$unknownR3.source_role = "unknown"
$unknownR3.source_effort = "unknown"
$unknownR3.dispatch_reason = "task_routing"
$unknownR3.routing_basis = "dispatcher_classification"
$unknownR3.routing_basis_ref = "none"

$typedWithoutReference = Copy-Request $unknownTyped
$typedWithoutReference.routing_basis_ref = "none"

$identityTupleMismatch = Copy-Request $r0
$identityTupleMismatch.source_role = "unknown"
$identityTupleMismatch.source_effort = "unknown"

$results = @()
$results += Invoke-Case -Name "r0_luna" -Request $r0 -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "r1_luna" -Request $r1 -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "r2_terra" -Request $r2 -ShouldPass $true -ExpectedModel "gpt-5.6-terra"
$results += Invoke-Case -Name "r3_sol" -Request $r3 -ShouldPass $true -ExpectedModel "gpt-5.6-sol"
$results += Invoke-Case -Name "unknown_classified_r0_luna" -Request $unknownClassifiedR0 -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "unknown_typed_r0_luna" -Request $unknownTyped -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "unknown_classified_r1_luna" -Request $unknownR1 -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "unknown_classified_r2_terra" -Request $unknownR2 -ShouldPass $true -ExpectedModel "gpt-5.6-terra"
$results += Invoke-Case -Name "unknown_classified_r3_sol" -Request $unknownR3 -ShouldPass $true -ExpectedModel "gpt-5.6-sol"
$results += Invoke-Case -Name "same_role_r0_luna" -Request $sameRole -ShouldPass $true -ExpectedModel "gpt-5.6-luna"
$results += Invoke-Case -Name "stale_rules" -Request $stale -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "incomplete_r1" -Request $incompleteR1 -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "mismatched_r1_evidence" -Request $mismatchedR1 -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "unbound_r1_evidence" -Request $unboundR1 -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "legacy_r1_lifecycle_tool" -Request $legacyR1 -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "under_role_r2" -Request $underRoleR2 -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "wsl_workspace_transport" -Request $wslWorkspace -ShouldPass $true -ExpectedModel "gpt-5.6-terra"
$results += Invoke-Case -Name "wsl_workspace_transport_contract_rejected" -Request $wslWorkspaceCloud -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "typed_without_reference" -Request $typedWithoutReference -ShouldPass $false -ExpectedModel ""
$results += Invoke-Case -Name "identity_tuple_mismatch" -Request $identityTupleMismatch -ShouldPass $false -ExpectedModel ""

[ordered]@{
    status = "PASS"
    rules_commit = $rulesCommit
    repository_head = $headCommit
    model_calls = 0
    cases = $results
} | ConvertTo-Json -Depth 8
Write-Output "DISPATCHER_DRY_RUN_TESTS_OK"
