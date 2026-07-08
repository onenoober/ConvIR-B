# v1 A0 Eval Tool Mismatch Note

Date: 2026-07-08

State: `FAILED_COMMAND`, repaired before scientific interpretation.

The first A0 val600 launch used `experience_docx/tools/eval_haze4k_checkpoint_compare.py`.
On the official anchor, `Dehazing/ITS/data/data_load.py::test_dataloader` does
not accept `depth_cache_dir`, `split_json`, or `split_name`, so the command
failed before inference with:

```text
TypeError: test_dataloader() got an unexpected keyword argument 'depth_cache_dir'
```

No metric result was produced, and no scientific result is interpreted from that
failed command. The repair is a narrow v1-only A0 evaluator that reads the
locked `val_inner` hazy filenames directly from
`haze4k_internal_split_2400_600.json` and does not touch Haze4K locked test.
