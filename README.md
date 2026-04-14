# Video Processing Pipeline (Python + OpenCV)

This repository provides a simple, memory-safe video processing pipeline for long recordings.
It reads video frames as a stream (never loading full files into RAM), computes a pixel-based motility metric, and writes results to an output folder.

## What It Does

- Reads one or many videos from a configured input path
- Supports long video files through frame-by-frame streaming with OpenCV
- Optionally uses logical segment windows (for example, every 10 minutes)
- Samples frames by either:
  - every N frames, or
  - every N seconds
- Computes motion between consecutive sampled frames using pixel differences
- Produces descriptive motility statistics for each segment and each video
- Writes JSONL details and a final JSON report to the configured output directory

## Out of Scope (Intentionally)

- Databases, ORMs, and queues
- Docker and CI/CD setup
- Cloud integrations
- Plugin architecture and abstract class hierarchies
- Unit test scaffold

## Requirements

- Python 3.10+
- OpenCV dependencies for your OS

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

All tuneable parameters are in [config.yaml](config.yaml):

- input/output paths
- segment duration
- frame sampling settings
- motility parameters (difference threshold, blur kernel, active-frame threshold)
- logging level and progress update interval

Sampling is strict: configure **exactly one** of:

- `sampling.every_n_frames`
- `sampling.every_n_seconds`

For pixel-based motility, the recommended setting is `sampling.every_n_frames: 1`.

## Run

```bash
python -m src.video_pipeline.cli --config ./config.yaml
```

## Output

For each input video, the pipeline writes one `.jsonl` file into the configured output directory.
Each line is a JSON object representing either:

- a sampled frame result with motion fields, or
- a segment summary with descriptive motility stats

The pipeline also writes `motility_report.json` with per-video and per-segment summaries.
