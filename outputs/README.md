# Outputs Directory

Use this ignored directory for generated experiment artifacts. Current runs use:

```text
outputs/
  mrrt/             MRRT target caches and four-arm single-step sweeps.
  msp_lvkl_*/        Official-low-variance-KL multi-step factorial runs.
  analysis/          JSON gates, paired analyses, and audited summaries.
  ckpt/              Saved post-training checkpoints.
```

Large generated files are ignored by git.
