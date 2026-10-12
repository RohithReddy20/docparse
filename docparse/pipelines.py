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
    # Same settings as parse-bench's built-in `infinity_parser2_flash`
    # (doc2json → bboxes), endpoint from INFINITY_ENDPOINT_URL.
    PipelineSpec(
        pipeline_name="infinity_parser2_flash_modal",
        provider_name="infinity_parser2_modal",
        product_type=ProductType.PARSE,
        config={
            "model_name": "infly/Infinity-Parser2-Flash",
            "task_type": "doc2json",
            "output_format": "json",
            "max_new_tokens": 16384,
        },
    ),
    # PP-DocLayoutV3 via serve/modal_paddle_layout.py; endpoint from PADDLE_LAYOUT_ENDPOINT_URL.
    PipelineSpec(
        pipeline_name="ppdoclayout_v3_modal",
        provider_name="paddle_layout_modal",
        product_type=ProductType.LAYOUT_DETECTION,
        config={"dpi": 200},
    ),
]

for _spec in _SPECS:
    register_pipeline(_spec)
