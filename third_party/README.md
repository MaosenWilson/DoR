# Third-Party Code

Upstream repositories are cloned here for reproducible external-platform
adapters. Their source, checkpoints, datasets, and environments are ignored by
this repository; this file records the exact provenance instead.

| Upstream | Pinned commit | License | DoR integration boundary |
| --- | --- | --- | --- |
| [RLVR-World](https://github.com/LaVi-Lab/RLVR-World) | local server checkout | upstream license | Reference implementation only. |
| [iVideoGPT](https://github.com/thuml/iVideoGPT) | `d601d5cac9e96c6aa0c17cb37ed6a7c7ca1fb210` | MIT | VP2-RoboSuite adapter may call the installed upstream package, but DoR owns reward, roll-out, and evaluation code. |
| [IRIS](https://github.com/eloialonso/iris) | `24326aaaa283c527f42b89b44cfdecf2665a7a16` | GPL-3.0 | Kept process-isolated. DoR must not import, copy, or link IRIS source into `src/dor`; it exchanges saved roll-outs only. |

Expected local layout:

```text
third_party/
  RLVR-World/
  iVideoGPT/
  iris/
```

The matching external weights and datasets live on the GPU server under
`/root/autodl-tmp/external_wm/`; they are deliberately not stored in this git
repository.
