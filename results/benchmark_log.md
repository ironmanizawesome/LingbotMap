# Resolution benchmark log

> outside_lab.MOV, forward 0–40s (turnaround at 40s). RTX 5080 (16 GB / 17.1 reported), SDPA + CPU offload.
> **Note:** 644/728 peak (18.5 / 22.4 GB) **exceeded the 16 GB VRAM and spilled into shared system RAM**
> (Windows CUDA sysmem fallback) — that spill, on top of SDPA + per-frame offload, is why they took
> 44–79 min. Only 518 (13.2 GB) stayed fully in VRAM. To keep higher res practical: FlashInfer (drop
> `--use_sdpa`), fewer frames (`--first_k`), `camera_num_iterations=1`, or a smaller sliding window.

| time | scene | frames | fps | img_size | WxH | peak_GB | infer_s | fps_eff | offload | sdpa | native | keyframe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-04 16:53 | outside_lab | 400 | 10 | 518 | 518x294 | 13.17 | 1899.6 | 0.2 | Y | Y | 518 | 1 |
| 2026-06-04 17:38 | outside_lab | 400 | 10 | 644 | 644x364 | 18.47 | 2648.0 | 0.2 | Y | Y | 518 | 1 |
| 2026-06-04 18:58 | outside_lab | 400 | 10 | 728 | 728x406 | 22.42 | 4730.8 | 0.1 | Y | Y | 518 | 1 |
