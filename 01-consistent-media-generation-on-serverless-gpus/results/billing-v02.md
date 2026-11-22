# Billed cost of the six v02 runs (Nebius billing export, November 2026)

Source: Nebius Billing one-time export (FOCUS 1.2 CSV); rows matched to each Job's compute
instance and boot disk. Machine = GPU preset (GPU + vCPU + RAM); disk = 200 GiB network SSD.
Region eu-north1, preemptible capacity, 21 Nov 2026 (UTC). Same numbers as the README tables.

| Model | GPU | Paid USD/h | Preparation: time, USD | Work, 10 images: time, USD | Total USD | USD per image |
|---|---|---|---|---|---|---|
| Sana | L40S | 0.7492 | 4.2 min, 0.0537 | 13.7 s, 0.0028 | 0.0566 | 0.00566 |
| Sana | H100 | 1.09 | 4.3 min, 0.0807 | 6.3 s, 0.0019 | 0.0826 | 0.00826 |
| Z-Image-Turbo | L40S | 0.7492 | 10.8 min, 0.1386 | 62.0 s, 0.0129 | 0.1515 | 0.01515 |
| Z-Image-Turbo | H100 | 1.1607 (1.09 → 1.39) | 7.9 min, 0.1554 | 21.1 s, 0.0068 | 0.1622 | 0.01622 |
| Qwen-Image | L40S | 0.7492 | 20.4 min, 0.2658 | 709.9 s, 0.1477 | 0.4135 | 0.04135 |
| Qwen-Image | H100 | 1.39 | 12.0 min, 0.2829 | 228.6 s, 0.0883 | **0.3712** | 0.03712 |

Definitions (identical in every table):
- Paid USD/h = billed machine cost / billed machine hours. L40S preemptible is billed per component:
  GPU 0.65 + 8 vCPU × 0.006 + 32 GiB × 0.0016 = 0.7492. H100 follows a spot price (the preset is billed as
  one line); for Z-Image the price changed from 1.09 to 1.39 inside the run, so the effective price is shown.
- c = paid USD/h × median generation seconds / 3600; work = 10 × c (time: 10 × median).
- Preparation = billed total (machine + disk) − work: VM start, container start, package install, weight
  download, model load, saving and shutdown. USD per image = total / 10.

Total for the six runs: USD 1.2376. Billing starts when the VM starts, not when the Job is created.
The `run-summary.json` files in `runs/` were written right after each run with the published H100 spot
minimum (USD 0.79/h); this file has what was actually paid.
