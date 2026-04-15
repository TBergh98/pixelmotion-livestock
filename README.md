# Video Motility Pipeline

Minimal pipeline (Python + OpenCV) to process long videos as a stream, compute pixel-based motility, and export results.

## What It Produces

- One JSONL file per video: frame-level records + segment summaries.
- One final report: data/output/motility_report.json with per-segment and per-video aggregates.

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

## Key Configuration

Main settings are in [config.yaml](config.yaml):

- segmentation.segment_duration_seconds
- sampling.every_n_frames or sampling.every_n_seconds (set exactly one)
- processing.diff_threshold
- processing.blur_kernel_size
- processing.active_motion_threshold

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
