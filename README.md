# Video Motility Pipeline

Pipeline Python + OpenCV per elaborare video lunghi in streaming, calcolare motilita pixel-based e produrre report, metriche descrittive e plot intra/inter-day.

## What It Produces

- Un JSONL per video con record frame-level e summary per segmento.
- Un report finale: data/output/motility_report.json.
- Se analytics e attivo: metriche e plot (PNG/HTML) separati per gruppo e giorno.
- Checkpoint incrementali per resume affidabile su run lunghi.

## Quick Start

Requirements:
- Python 3.10+
- OpenCV runtime dependencies for your OS

Create and activate Conda environment:

```bash
conda env create -f environment.yml
conda activate pixelmotion-livestock
```

Alternative (already active environment):

```bash
pip install -r requirements.txt
```

Run:

```bash
python -m src.video_pipeline.cli --config ./config.yaml
```

## Input Hierarchy (Optional but recommended)

Per analisi multi-gruppo/multi-giorno, usa una struttura cartelle gerarchica:

```text
data/input/
	GroupA/
		2026-04-14/
			video_01.mp4
		2026-04-15/
			video_02.mp4
	GroupB/
		2026-04-14/
			video_01.mp4
```

Con hierarchy.enabled=true, la pipeline estrae automaticamente:
- group_id dalla cartella gruppo
- recording_date dalla cartella giorno (ISO YYYY-MM-DD)

Se fail_on_mixed_groups=true, il run fallisce quando trova gruppi multipli nello stesso run non segregato.

## Output Structure

La pipeline salva ora output divisi in cartelle per gruppo e giorno:

```text
data/output/
	results/
		GroupA/
			2026-04-15/
				video_02_results.jsonl
		GroupB/
			2026-04-14/
				video_01_results.jsonl
	analytics/
		GroupA/
			2026-04-15/
				intraday_metrics.json
				plots/
					intraday_timeseries_2026-04-15_GroupA.png
					intraday_timeseries_2026-04-15_GroupA.html
					intraday_distribution_2026-04-15_GroupA.png
					intraday_distribution_2026-04-15_GroupA.html
					spatial_heatmap_2026-04-15_GroupA.png
			interday/
				interday_metrics.json
				plots/
					interday_trend_GroupA.png
					interday_trend_GroupA.html
					interday_delta_GroupA.png
					interday_delta_GroupA.html
	motility_report.json
```

Nessun confronto cross-group viene prodotto: ogni gruppo resta isolato per design.

## Key Configuration

Main settings are in [config.yaml](config.yaml):

- segmentation.segment_duration_seconds
- sampling.every_n_frames or sampling.every_n_seconds (set exactly one)
- processing.diff_threshold
- processing.blur_kernel_size
- processing.active_motion_threshold

Spatial heatmap settings (processing):
- processing.compute_spatial_grid (default: false) — Enable spatial motion tracking
- processing.spatial_grid_size (default: 16) — Grid granularity (e.g., 16 = 16×16 = 256 cells per frame)

Analytics/hierarchy settings:
- hierarchy.enabled
- hierarchy.levels
- hierarchy.fail_on_mixed_groups
- analytics.enabled
- analytics.intraday_window_seconds
- analytics.output_formats.png
- analytics.output_formats.html
- analytics.plots.intraday_timeseries
- analytics.plots.intraday_distribution
- analytics.plots.intraday_heatmap — Spatial heatmap showing motion concentration
- analytics.plots.interday_trend
- analytics.plots.interday_delta

Checkpoint settings:
- checkpoint.enabled
- checkpoint.directory
- checkpoint.save_every_frames
- checkpoint.validate_config_snapshot
- checkpoint.strict_resume

## Checkpoint and Resume

Per run lunghi, la pipeline salva checkpoint incrementali (atomici) durante l'elaborazione video:

- checkpoint periodico ogni `checkpoint.save_every_frames`
- checkpoint a ogni cambio segmento
- checkpoint finale con stato `video_completed`

Alla ripartenza:

- se trova un checkpoint `running`, ripristina stato statistico/sampling e riprende dal frame successivo
- se trova un checkpoint `video_completed`, salta il video gia concluso
- in `strict_resume=true`, la pipeline fallisce su mismatch critici (config snapshot o file output incoerenti)

## How Metrics Are Calculated

The pipeline compares each sampled frame with the previous sampled frame (inside the same segment).

1. Pre-processing
- Convert both frames to grayscale.
- Optionally apply Gaussian blur (kernel = blur_kernel_size).

2. Pixel difference and threshold
- Compute absolute difference image.
- Mark pixel as changed if diff >= diff_threshold.

3. Frame-level metrics
- changed_pixels = number of changed pixels
- total_pixels = frame width * frame height
- active_pixel_ratio = changed_pixels / total_pixels
- motility_score = active_pixel_ratio
- mean_diff_intensity = mean(abs_diff) / 255

Formally:

$$
m = \frac{C}{P}
$$

$$
d = \frac{\mathrm{avg}(|I_t - I_{t-1}|)}{255}
$$

where $m$ is motility score, $C$ is changed pixels, $P$ is total pixels, and $d$ is mean diff intensity.

4. Segment/video aggregate metrics
- count: number of valid motility scores in the group
- mean: average of motility_score
- std: standard deviation
- min, max: extrema of motility_score
- active_ratio: fraction of scores >= active_motion_threshold

Formally:

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
$$

$$
\sigma = \sqrt{\max\left(\frac{1}{N}\sum_{i=1}^{N}x_i^2 - \mu^2,\,0\right)}
$$

$$
r = \frac{|\{x_i \mid x_i \geq \tau\}|}{N}
$$

where $x_i$ is motility score, $\tau$ is active motion threshold, and $r$ is active ratio.

## Spatial Heatmap

When `processing.compute_spatial_grid=true`, the pipeline additionally computes spatial motion density during frame processing:

1. **Grid Computation (per frame pair)**
   - After computing the binary thresholded difference image, divide it into a `spatial_grid_size × spatial_grid_size` grid (e.g., 16×16 = 256 cells)
   - For each cell: count changed pixels and compute density = (changed_pixels / total_pixels_in_cell)
   - Store the flattened grid (256 values) in the JSONL record for each frame

2. **Spatial Aggregation (per intra-day window)**
   - During analytics, for each intra-day time window (e.g., 5 minutes), load all spatial grids from frames in that window
   - Sum the density values across all frames in the window
   - Normalize by dividing by frame count to get average density per cell: avg_cell_density = sum / frame_count
   - Result: aggregated grid for the window

3. **Heatmap Visualization**
   - Average all window grids across the entire day
   - Reshape flattened grid back to 2D (`spatial_grid_size × spatial_grid_size`)
   - Generate PNG heatmap with matplotlib, using "hot" colormap (red = high motion density, blue = low)
   - Add grid overlay to show cell boundaries and colorbar for density scale

**Use case**: Identify where animals concentrate in the frame and where they move most, enabling localized behavior analysis.

**Performance**: Grid computation is relatively efficient (done during main video processing). Storage overhead is ~2-4 KB per frame (~256 floats). Can be disabled with `compute_spatial_grid=false` to reduce JSONL size if not needed.

## Notes

- The first sampled frame of each segment has no previous frame to compare against, so it has no motility fields.
- For this reason, sampled_frames can be greater than motility.count.
- Se analytics.enabled=true ma matplotlib/plotly non sono disponibili, la pipeline continua e logga warning/error sui plot mancanti.
- Se spatial_grid=false, i record JSONL non includeranno dati spaziali e le heatmap non verranno generate.
