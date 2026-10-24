# Changelog

## Repo Versions

Tracks changes of the codebase. 

### 0.8.0 - (2026-10-24)

* Added invocation of a few benchmarks as unit tests.
* All docker containers now optionally have a "downloads" directory for saving assets outside the 
container. CLI arg for this as well.
* Underlying script for llama bench `run_llama_bench.sh` updated to add more models, and updated
running mechanism to run multiple models in a sequence.
* Added a ton of new model generation benchmarks. New models, new sizes, new quants, new context
windows.
* Added new benchmark, `llama_bench_max_model_parameters_q4_k_m` to see the largest number of LLM
parameters that can be loaded into the attached GPUs using real models.
* New flagging code to force rebuilding docker images.
* Added a new field `gpu_power_usage` (and type `ReportFileGPUPowerUsage`) to `BenchmarkResult`
that records GPU energy consumption for a given benchmark.
* Added a new visualization mode `compare-llms` specifically to look at LLM performance when varying
quantization, context depth, model and system build. The `compare-llm-families` does the same thing
but creates a viz for all the families in the input files.
* Added several `llama-bench` tests that use `--tensor-split` across attached GPUs per feedback from
first post.
* Fixed bug in sequence execution logic that could lead to false failures.
* Added ability to group systems in visualizations, they have similar colors to make seeing groups
easier.
* Brought out more visualization params to CLI.

### 0.7.0 - (2026-07-11)

* Added `--x-label` and `--labeled-input` to comparison mode to be able to have more control over
how plot axes are labeled.
* Added `--only-test` and a few other CLI options to comparison mode to limit visualization plots.
* Introduced a more exhaustive set of unit tests around rendering figures to ensure edge cases
at least produce images.
* Switched `benchmark_name` type in result to the actual Enum instead of a string.
* Improved docs, both in the README and CLI. 
* Flattened type system to a single file.

### 0.6.0 - (2026-05-07)

* Bumped a few underlying test versions to be able to support Tesla T40.
* Added a CLI flag to be able to optionally flatten down visualization version.

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

Tracks changes of the main output file. There are per-test version histories in the python wrappers.

### 0.3.0 - (2026-10-24)

* Introduced new idiom to `ReportFileNumerical`. If a benchmark fails on ANY individual GPU but is 
able to finish in the multi GPU case, `min_by_gpu_type`, `max_by_gpu_type`, `mean_by_gpu_type`, 
`theoretical_multi_gpu_mean`, `theoretical_multi_gpu_sum`, `forced_multi_gpu_numerical_mean` will
all be set to None. 
* Added a new field `gpu_power_usage` (and type `ReportFileGPUPowerUsage`) to `BenchmarkResult`
that records GPU energy consumption for a given benchmark.
* Added new field to output to record tests that failed.

### 0.2.0 - (2026-05-07)

* Bumped version of Blender, and updated Llama bench to be able to support Tesla T40

### 0.1.0 - (2026-01-18)

* Initial implementation and format.