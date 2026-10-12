"""`docparse` = `parse-bench` with docparse providers/pipelines pre-registered.

Usage mirrors parse-bench, e.g.:
    docparse run lightonocr3_0_8b_grounding --test
    docparse pipelines
"""

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    import docparse.pipelines  # noqa: F401
    from parse_bench.cli import main as parse_bench_main

    return parse_bench_main()
