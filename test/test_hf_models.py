"""
Tests to make sure HF models exist and our wrapper code works as expected.
"""

from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench import hf_models
from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench.llama_bench_types import (
    HFModelSpec,
)
from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench import (
    llama_bench_dockerfile_wrapper,
)


def test_models_exist() -> None:
    """
    Traverses the hf_models module to find every HFModelSpec instance defined,
    regardless of whether it's directly exposed or nested inside ModelQuantization objects.

    This ensures that all models referenced by our benchmark configurations
    actually exist on HuggingFace.
    :return: None
    """

    found_specs = []

    def _collect_specs(obj: object) -> None:
        """
        Recursively collects HFModelSpec instances from an object.
        Output is the contents of `found_specs`
        :param obj: To scan.
        :return: None
        """

        if isinstance(obj, HFModelSpec):
            found_specs.append(obj)
            return

        # Handle Pydantic models (like ModelQuantization) which have __dict__ for fields
        if hasattr(obj, "__dict__"):
            for key, val in obj.__dict__.items():
                if not key.startswith("_"):  # Ignore private attributes
                    _collect_specs(val)

        # Handle lists/tuples of specs (e.g., MODEL_SIZE_Q4_SPECS)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect_specs(item)

    # Iterate through all public variables in the hf_models module
    for name in dir(hf_models):
        if not name.startswith("_"):  # Ignore private variables/magic methods
            val = getattr(hf_models, name)
            _collect_specs(val)

    # Deduplicate specs by object ID to avoid checking the same model multiple times
    seen_ids = set()
    unique_specs = []
    for spec in found_specs:
        obj_id = id(spec)
        if obj_id not in seen_ids:
            unique_specs.append(spec)
            seen_ids.add(obj_id)

    # Run the validation logic from our wrapper code
    llama_bench_dockerfile_wrapper.verify_hf_model_specs(unique_specs)
