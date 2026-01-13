# GPU Box Benchmark - gpu_box_benchmark 

![](./art.png)

A set of benchmarks selected to compare different GPU server builds.

Benchmarks are defined as dockerfiles and run as docker containers for portability. The outer
python code is responsible for configuring, running and collecting the results from the internal
containers.

## Usage

After creating a virtual environment, it should be easy to start running the benchmarks as they're
just a series of CLIs with sensible defaults.

```
(.venv) devon@ESO-3-DEV-VM:~/Documents/bray_airways/gpu_box_benchmark$ python bench_cli.py --help
Usage: bench_cli.py [OPTIONS] COMMAND [ARGS]...

  Uses Docker to run GPU related benchmarks with the goal of comparing system
  hardware architectures.

Options:
  --help  Show this message and exit.

Commands:
  benchmark           Run one or more benchmarks and records the results.
  explain-benchmarks  Prints a description about what each of the supported
                      benchmarks do.
(.venv) devon@ESO-3-DEV-VM:~/Documents/bray_airways/gpu_box_benchmark$ python bench_cli.py benchmark --help
Usage: bench_cli.py benchmark [OPTIONS]

  Run one or more benchmarks and records the results.

  :param test: See click help for docs! :param gpu: See click help for docs!
  :param output_path: See click help for docs! :param title: See click help
  for docs! :param description: See click help for docs! :return: None

Options:
  --test TEXT         Decides which benchmark to run.
                      Options below. Either provide index or value:
                         0: resnet50_train_batch_1_amp
                         1: resnet50_train_batch_64_amp
                         2: resnet50_infer_batch_1_amp
                         3: resnet50_infer_batch_256_amp  [default: resnet50_infer_batch_256_amp]
  -g, --gpu [0|1]     The GPU(s) to use for computation. Can be given multiple
                      times. If not given, all GPUs will be used.
  --output-path FILE  The resulting system evaluation is written to this path.
                      [default: /home/devon/Documents/bray_airways/gpu_box_ben
                      chmark/benchmark_result.json]
  --title TEXT        A short few words to describe the run. Will be a plot
                      title in resulting visualizations.  [default: Sample
                      Title]
  --description TEXT  Longer text field to qualitatively describe the run in a
                      more verbose way.  [default: Sample Description. This
                      run was completed on a computer made of corn!]
  --help              Show this message and exit.
```

## Getting Started

### Python Dependencies

Poetry is required to manage Python dependencies. You can install it easily by following the
operating system specific instructions [here](https://python-poetry.org/docs/#installation).

`pyproject.toml` contains dependencies for required Python modules for building, testing, and 
developing. They can all be installed in a [virtual environment](https://docs.python.org/3/library/venv.html) 
using the follow commands:

```
python3.10 -m venv .venv
source ./.venv/bin/activate
poetry install
```

There's also a bin script to do this, and will install poetry if you don't already have it:

```
./tools/create_venv.sh
```

## Developer Guide

The following is documentation for developers that would like to contribute
to GPU Box Benchmark.

### Pycharm Note

Make sure you mark `gpu_box_benchmark` and `./test` as source roots!

### Testing

This project uses pytest to manage and run unit tests. Unit tests located in the `test` directory 
are automatically run during the CI build. You can run them manually with:

```
./tools/run_tests.sh
```

### Local Linting

There are a few linters/code checks included with this project to speed up the development process:

* Black - An automatic code formatter, never think about python style again.
* Isort - Automatically organizes imports in your modules.
* Pylint - Check your code against many of the python style guide rules.
* Mypy - Check your code to make sure it is properly typed.

You can run these tools automatically in check mode, meaning you will get an error if any of them
would not pass with:

```
./tools/run_checks.sh
```

Or actually automatically apply the fixes with:

```
./tools/apply_linters.sh
```

There are also scripts in `./tools/` that include run/check for each individual tool.


### Using pre-commit

Upon cloning the repo, to use pre-commit, you'll need to install the hooks with:

```
pre-commit install --hook-type pre-commit --hook-type pre-push
```

By default:

* black
* pylint
* isort
* mypy

Are all run in apply-mode and must pass in order to actually make the commit.

Also by default, pytest needs to pass before you can push.

If you'd like skip these checks you can commit with:

```
git commit --no-verify
```

If you'd like to quickly run these pre-commit checks on all files (not just the staged ones) you
can run:

```
pre-commit run --all-files
```


