# CHD-RM v2b Need Target Transform Definitions

D6a quantile target maps raw need through the train_inner empirical CDF.
D6b log target uses log1p(raw_need / train_median_raw_need), then train_inner p1/p99 normalization.
D6c ordinal target uses the quantile target and BCE labels at q20/q33/q66/q80.
All statistics are computed from train_inner only. Locked Haze4K test is not used.
