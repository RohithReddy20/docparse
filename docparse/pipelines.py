"""docparse pipeline specs, registered into parse-bench's registry on import."""

from parse_bench.inference.pipelines import register_pipeline
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType

import docparse.providers  # noqa: F401  (registers providers)

_SPECS = [
    PipelineSpec(
        pipeline_name="lightonocr3_0_8b_grounding",
        provider_name="lightonocr3",
        product_type=ProductType.PARSE,
        config={"prompt_mode": "grounding"},
    ),
    PipelineSpec(
        pipeline_name="lightonocr3_0_8b_md",
        provider_name="lightonocr3",
        product_type=ProductType.PARSE,
        config={"prompt_mode": "transcribe"},
    ),
]

for _spec in _SPECS:
    register_pipeline(_spec)
