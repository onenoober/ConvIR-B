# v2f First-Stage Summary

- LDHN core fraction of LDHN: `0.569798970635499`
- LDHN boundary fraction of LDHN: `0.43020102936450094`
- Best feature probe: `{'threshold': 0.4924335181713104, 'positive_prevalence': 0.5, 'coverage': 0.5, 'precision': 0.7270882290108951, 'recall': 0.7270882290108951, 'false_positive_rate': 0.2729117709891049, 'auroc': 0.8107264347671554, 'auprc': 0.807792756659645, 'feature_set': 'feature_set_2', 'probe': 'mlp', 'train_rows': 74996, 'val_rows': 18724}`
- Locked Haze4K test usage: `none`

Decision: F4 density-stratified frozen-side head canary is authorized. It must
keep ConvIR-B and D3 frozen and report the original v2e global LDHN/false-tail
gate. It does not authorize D2, v3, RARM, or locked Haze4K test.
