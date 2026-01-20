# GPU Box Benchmark - gpu_box_benchmark 

![](./art.png)

A set of benchmarks selected to compare different GPU server builds. Uses docker to force parallelization of normally single GPU tests.

Benchmarks are defined as dockerfiles and run as docker containers for portability. The outer
python code is responsible for configuring, running and collecting the results from the internal
containers.

## Usage

After creating a virtual environment, it should be easy to start running the benchmarks as they're
just a series of CLIs with sensible defaults.

```
$ python bench_cli.py --help
Usage: bench_cli.py [OPTIONS] COMMAND [ARGS]...

  Uses Docker to run GPU related benchmarks with the goal of comparing system
  hardware architectures.

Options:
  --help  Show this message and exit.

Commands:
  benchmark           Run one or more benchmarks and records the results.
  explain-benchmarks  Prints a description about what each of the supported
                      benchmarks do.
  render-systemd      Creates a systemd unit that will start the run at boot.
```

The `render-systemd` command exists to create a systemd unit that runs the full benchmarking suite
at boot. This is useful for working through a large number of hardware combinations because all
you have to do is start the PC to run the benchmarking suite. Output can be written anywhere
including a NAS. 

## Benchmarks


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


