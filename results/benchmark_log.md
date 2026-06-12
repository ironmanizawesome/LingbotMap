# Resolution benchmark log

> outside_lab.MOV, forward 0–40s (turnaround at 40s). RTX 5080 (16 GB / 17.1 reported), SDPA + CPU offload.
> **Note:** 644/728 peak (18.5 / 22.4 GB) **exceeded the 16 GB VRAM and spilled into shared system RAM**
> (Windows CUDA sysmem fallback) — that spill, on top of SDPA + per-frame offload, is why they took
> 44–79 min. Only 518 (13.2 GB) stayed fully in VRAM. To keep higher res practical: FlashInfer (drop
> `--use_sdpa`), fewer frames (`--first_k`), `camera_num_iterations=1`, or a smaller sliding window.

## Progress notes

**2026-06-12 — 644 settled; `compare_resolution.py` → `precompute_npz.py` (renamed via `git mv`).**
The script is now the "run inference once → save npz" producer stage that feeds LAI (downstream
`map-LAIcrop/pipeline.py` will read the npz instead of re-running the model).

**2026-06-12 — 3D-RoPE 1024-frame crash fixed.** With `--keyframe_interval 1`, the RoPE time
index (`total_frames_processed`) advances once per frame and hit the precomputed RoPE table's
fixed length (`max_frame_num`, previously hardcoded to 1024). At frame 1024 the time-axis lookup
ran off the table and RoPE collapsed (head dim 32→22) → crash in `apply_rotary_emb`. Latent until
now: every 644 run above had ≤965 frames; the 1058-frame `Watermelon_fullbed_sideview` was the
first to exceed 1024 (crashed at 1024/1058 after 2h39m). **Fix:** `--max_frame_num` is now a CLI
arg (default 1024, mirrors demo.py). **For any clip >1024 frames, pass `--max_frame_num` above the
frame count (e.g. 1200).** Raising it leaves frames 0–1023 bit-identical (the table is
arange-indexed) and only extends the tail, so earlier npz files are unaffected.

| time | scene | frames | fps | img_size | WxH | peak_GB | infer_s | fps_eff | offload | sdpa | native | keyframe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-04 16:53 | outside_lab | 400 | 10 | 518 | 518x294 | 13.17 | 1899.6 | 0.2 | Y | Y | 518 | 1 |
| 2026-06-04 17:38 | outside_lab | 400 | 10 | 644 | 644x364 | 18.47 | 2648.0 | 0.2 | Y | Y | 518 | 1 |
| 2026-06-04 18:58 | outside_lab | 400 | 10 | 728 | 728x406 | 22.42 | 4730.8 | 0.1 | Y | Y | 518 | 1 |
| 2026-06-08 18:07:24 | highup_plant | 40 | 10 | 644 | 644x364 | 10.63 | 12.6 | 3.2 | Y | Y | 518 | 1 |
| 2026-06-09 15:06:45 | highup_plant | 800 | 10 | 644 | 644x364 | 20.04 | 8761.7 | 0.1 | Y | Y | 518 | 1 |
| 2026-06-10 17:28:43 | Strawberry_onebed | 705 | 10 | 644 | 644x364 | 19.66 | 4004.9 | 0.2 | Y | Y | 518 | 1 |
| 2026-06-10 18:10:35 | Strawberry_oneside | 329 | 10 | 644 | 644x364 | 18.17 | 1909.4 | 0.2 | Y | Y | 518 | 1 |
| 2026-06-10 19:34:36 | Watermelon_fullbed_straightview | 965 | 10 | 644 | 644x364 | 20.70 | 4634.8 | 0.2 | Y | Y | 518 | 1 |
