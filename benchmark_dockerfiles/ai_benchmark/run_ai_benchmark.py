"""
Runs inside the ai-benchmark docker container.
This is not nice looking code! IT's gotta be COMPACT!
"""

import logging
import warnings
import os

# Disable all tensorflow info/debug/warning.  0 = all logs, 1 = info, 2 = warning, 3 = error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress Python warnings globally ----
warnings.filterwarnings("ignore")

# Suppress logging from imported modules ----
logging.getLogger().setLevel(logging.CRITICAL)

from ai_benchmark import AIBenchmark

# Fully silent, GPU only
benchmark = AIBenchmark(use_CPU=False, verbose_level=0)

# Run full benchmark (training + inference)
results = benchmark.run()

# Write output (a float) directly to stdout so it can be read in by the docker monitor.
print(results.ai_score)
