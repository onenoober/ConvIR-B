# Command Control Hardening Acceptance

The candidate passed the isolated CPU-only cloud acceptance on convir-4090.
The final marker was `CONVIR_OPS_V432_CLOUD_OK` at candidate commit
`8447f8dd54b079505a24a8e6df8d1716256ba4b2`, with 111 tests, schema v4,
exactly six MCP tools, zero model calls, zero GPU access, and zero protected
data access.

The accepted changes bind writes to the requested worktree, replace fragile
cross-shell repository reads with fixed-argv ref-bound readers, provide one
generic workload-progress writer while accepting completed A1 NAME_PROGRESS
telemetry, and remove duplicated incident recipes from the default command
protocol. The MCP version, schema, tool count, scientific contracts, data
roles, gates, thresholds, seeds, budgets, and protected-data policy are
unchanged.

The two earlier validation failures were test-harness boundary defects only:
the remote shell cannot inherit a local branch environment, and the standalone
probe initially lacked PYTHONPATH. Both were corrected before the final marker;
the candidate implementation then passed the complete matrix.
