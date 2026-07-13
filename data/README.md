# Data Directory

This ignored directory is a server-side mount for the RT-1 `fractal20220817`
episodes used by the current experiments.

```text
data/
  processed/
    fractal20220817_data/
      *.npz
```

Each episode must contain `image` (`uint8`, `[T,H,W,3]`) and `action`
(`float`, `[T,13]`). The active loader is `src/dor/episodes.py`.
