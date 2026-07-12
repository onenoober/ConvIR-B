# v3o Data Mapping Contract

Input order, names, fold map, clean-reference groups, base checkpoint, control
checkpoint, D7c artifact, density artifact, frozen operator artifacts, and
fixed-alpha reference are inherited from the SHA-verified v3m A1 contract.

The route must reject any missing name, duplicate operator/name key, fold-map
drift, unexpected candidate count, or mismatch in the fixed-alpha reference.
The Haze4K locked test is forbidden.

`v3j_route_confirm` is supplied only as the inherited shared validator's
non-test split sentinel. v3o-A0 does not read that key, load its names, or use
route-confirm data for selection or metrics.
