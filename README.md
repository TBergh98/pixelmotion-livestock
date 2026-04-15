# Video Motility Pipeline

Pipeline Python + OpenCV per elaborare video lunghi in streaming, calcolare motilita pixel-based e produrre report, metriche descrittive e plot intra/inter-day.

## What It Produces

- Un JSONL per video con record frame-level e summary per segmento.
- Un report finale: data/output/motility_report.json.
- Se analytics e attivo: metriche e plot (PNG/HTML) separati per gruppo e giorno.

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
- analytics.plots.interday_trend
- analytics.plots.interday_delta

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

## Notes

- The first sampled frame of each segment has no previous frame to compare against, so it has no motility fields.
- For this reason, sampled_frames can be greater than motility.count.
- Se analytics.enabled=true ma matplotlib/plotly non sono disponibili, la pipeline continua e logga warning/error sui plot mancanti.
