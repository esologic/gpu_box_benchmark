# Changelog

## Repo Versions

Tracks changes of the codebase. 

### 0.5.0 - (2026-02-03)

* Expands `llama.cpp` benchmarks to include two MoE models.
* Adds `gdsio` benchmarking to measure how quickly data can be loaded from disk into VRAM.
* Adds a `hashcat` SHA-256 benchmark as another non-AI point of comparison.
* Adds ansible playbook to install known good GPU driver version, docker and nvidia-docker.
* All benchmarks now support Kepler -> Volta GPUs.

### 0.4.0 - (2026-01-25)

* Adds `compare` mode for visualizing difference between runs.
* Steps taken to reclaim disk space used by docker between runs.
* Cleanup features after running a few test benchmarks.
* Adds labels to all Dockerfiles for image cleanup. Corresponding CLI flag as well.
* Introduces per plot visualization metric.
* Improved viz clarity.
* Basic unit testing suite.

### 0.3.0 - (2026-01-20)

* Fills in implementation for `explain-benchmarks`. 
* Published repo to GitHub!

### 0.2.0 - (2026-01-20)

* Multi GPU native tests will also be run in parallelized mode to compare with native performance.
* Implements `render-systemd` command to run the complete benchmarking suite at boot. 
* Adds content aware timelapse benchmark to measure frames per second.
* Adds whisper throughput benchmark.
* Implements a mechanism to run single GPU benchmarks (FAHBench, Blender, Ai-B) on multiple GPUs
simultaneously using docker.
* Adds support for the `ai-benchmark` suite run in a container.
* Adds Folding@Home support via FAHBench.
* Adds `llama-bench` benchmarks for quantifying LLM performance.
* Adds the blender benchmark for a non AI reference point. 

### 0.1.0 - (2026-01-13)

* First working version. Adds a suite of ResNet50-related benchmarks.
* Defines input/output formats and repo structure.

### 0.0.1 - (2026-01-10)

* Project begins


## Format Versions

Tracks changes of the main output file.

### 0.1.0 - (2026-01-18)

* Initial implementation and format.