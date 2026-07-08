# V48 Train-Derived K-fold Tail-safe Protocol

Use Haze4K train only. Do not enumerate or evaluate Haze4K test. Build grouped folds by base image id and balance train-derived proxies. Train the fixed DCFSB-bottleneck adapter-only recipe per fold from official A0, then evaluate only the held-out fold against matched A0 on the same images.
