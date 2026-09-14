"""Tests for the detailed tool-trace parsers in run_benchmark.py.

The agent_run.log format (produced by Rich in churro/main.py) soft-wraps long
``args:`` payloads across several lines at word boundaries, so the parsers must
fold continuation lines back together before JSON-decoding. The wrapped
write_file fixture below mirrors the real logged format byte-for-byte.

This test lives in ``tests/`` (not ``benchmarks/``) so the root-wide pytest
collection does not descend into ``benchmarks/step19/baseline``, whose
known-broken fixture tests intentionally fail to import standalone.
"""
import sys
import os

BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks",
    "step19",
)
sys.path.insert(0, BENCHMARK_DIR)

from run_benchmark import (
    parse_tool_trace,
    parse_detailed_trace,
    derive_detailed_metrics,
    _consume_continuation,
    _unfold,
)

WRAPPED_ARGS = (
    '      args: {"content": "\\"\\"\\"A small shopping cart built on the pricing \n'
    'module.\\"\\"\\"\\n\\nfrom pricing import line_total\\n\\n\\nclass ShoppingCart:\\n    \n'
    'def __init__(self):\\n        self._items = []\\n\\n    def add_item(self, name: \n'
    'str, quantity: int, unit_price: float) -> None:\\n        \n'
    'self._items.append({\\"name\\": name, \\"quantity\\": quantity, \\"price\\": \n'
    'unit_price})\\n\\n    def item_count(self) -> int:\\n        return \n'
    'sum(item[\\"quantity\\"] for item in self._items)\\n\\n    def total(self) -> \n'
    'float:\\n        return round(\\n            sum(\\n                \n'
    'line_total(item[\\"quantity\\"], item[\\"price\\"])\\n                for item in \n'
    'self._items\\n            ),\\n            2,\\n        )\\n", "path": \n'
    '"shopping_cart.py"}'
)

REALISTIC_LOG = (
    "CHURRO\n"
    "> -- Agent --\n"
    "_thinking_ start\n"
    "[tool] list_files\n"
    '      args: {"recursive": true}\n'
    "[tool] run_tests (failed)\n"
    "      args: {}\n"
    "      error: Test suite executed but tests failed (pytest exit code 1)\n"
    "[tool] read_file\n"
    '      args: {"path": "shopping_cart.py"}\n'
    "[tool] read_file\n"
    '      args: {"path": "pricing.py"}\n'
    "[tool] read_file\n"
    '      args: {"path": "tests/test_shopping_cart.py"}\n'
    "[tool] write_file\n"
    + WRAPPED_ARGS
    + "\n"
    "[tool] run_tests\n"
    "      args: {}\n"
    "> Agent finished with no text response.\n"
    "> Session saved. Goodbye.\n"
    "\n"
    "---STDERR---\n"
)


def assert_eq(actual, expected, label=""):
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"


def test_parse_tool_trace_backward_compatible():
    trace = parse_tool_trace(REALISTIC_LOG)
    assert_eq(
        trace,
        [
            ("list_files", True),
            ("run_tests", False),
            ("read_file", True),
            ("read_file", True),
            ("read_file", True),
            ("write_file", True),
            ("run_tests", True),
        ],
        "parse_tool_trace",
    )


def test_parse_detailed_trace_simple_records():
    records = parse_detailed_trace(REALISTIC_LOG)
    assert_eq(len(records), 7, "record count")
    assert_eq(records[0], {"name": "list_files", "success": True, "args": {"recursive": True}, "error": ""})
    assert_eq(records[1]["name"], "run_tests", "name")
    assert_eq(records[1]["success"], False, "success")
    assert_eq(records[1]["error"], "Test suite executed but tests failed (pytest exit code 1)", "error")
    assert_eq(records[2], {"name": "read_file", "success": True, "args": {"path": "shopping_cart.py"}, "error": ""})


def test_parse_detailed_trace_wrapped_write_file():
    records = parse_detailed_trace(REALISTIC_LOG)
    wf = records[5]
    assert_eq(wf["name"], "write_file", "write_file name")
    assert_eq(wf["success"], True, "write_file success")
    assert_eq(wf["args"].get("path"), "shopping_cart.py", "write_file path extracted from wrapped JSON")
    content = wf["args"].get("content", "")
    assert '"A small shopping cart built on the pricing module."""' in content, "content reconstructed"
    assert "class ShoppingCart:" in content, "content middle"
    assert "item[\"quantity\"]" in content, "content tail"
    assert_eq(records[6]["name"], "run_tests", "record after wrapped block")
    assert_eq(records[6]["success"], True, "record after wrapped block success")


def test_derive_detailed_metrics():
    records = parse_detailed_trace(REALISTIC_LOG)
    m = derive_detailed_metrics(records)
    assert_eq(m["list_files_count"], 1, "list_files_count")
    assert_eq(m["search_files_count"], 0, "search_files_count")
    assert_eq(m["read_file_count"], 3, "read_file_count")
    assert_eq(m["write_file_count"], 1, "write_file_count")
    assert_eq(m["run_tests_count"], 2, "run_tests_count")
    assert_eq(
        m["files_inspected"],
        ["shopping_cart.py", "pricing.py", "tests/test_shopping_cart.py"],
        "files_inspected",
    )
    assert_eq(m["files_modified"], ["shopping_cart.py"], "files_modified")
    assert_eq(m["first_run_tests_index"], 1, "first_run_tests_index")
    assert_eq(m["run_tests_before_inspection"], True, "run_tests_before_inspection")
    assert_eq(m["reads_before_fix"], 3, "reads_before_fix")
    assert_eq(m["sequence_string"].count("x3"), 1, "multi reading collapsed")
    assert_eq(m["sequence_string"].count("->"), 4, "sequence joins")


def test_empty_inputs():
    assert_eq(parse_tool_trace(""), [], "empty trace")
    assert_eq(parse_detailed_trace(""), [], "empty records")
    m = derive_detailed_metrics([])
    assert_eq(m["files_inspected"], [], "empty inspected")
    assert_eq(m["files_modified"], [], "empty modified")
    assert_eq(m["first_run_tests_index"], -1, "empty first_rt")
    assert_eq(m["run_tests_before_inspection"], False, "empty rt_before_rf")
    assert_eq(m["sequence_string"], "", "empty seq")


def test_no_run_tests():
    log = (
        "[tool] read_file\n"
        '      args: {"path": "x.py"}\n'
        "[tool] write_file\n"
        '      args: {"content": "y", "path": "x.py"}\n'
    )
    m = derive_detailed_metrics(parse_detailed_trace(log))
    assert_eq(m["first_run_tests_index"], -1, "no rt idx")
    assert_eq(m["run_tests_before_inspection"], False, "no rt before")
    assert_eq(m["run_tests_count"], 0, "no rt count")


def test_no_write_file():
    log = (
        "[tool] read_file\n"
        '      args: {"path": "a.py"}\n'
        "[tool] run_tests\n"
        "      args: {}\n"
    )
    m = derive_detailed_metrics(parse_detailed_trace(log))
    assert_eq(m["reads_before_fix"], 1, "reads before fix")
    assert_eq(m["files_modified"], [], "no modified")


def test_duplicate_reads_deduped():
    log = (
        "[tool] read_file\n"
        '      args: {"path": "a.py"}\n'
        "[tool] read_file\n"
        '      args: {"path": "a.py"}\n'
        "[tool] read_file\n"
        '      args: {"path": "b.py"}\n'
    )
    m = derive_detailed_metrics(parse_detailed_trace(log))
    assert_eq(m["files_inspected"], ["a.py", "b.py"], "deduped inspected")
    assert_eq(m["read_file_count"], 3, "raw count")


def test_consume_continuation_stops_at_directives():
    lines = [
        "      args: {",
        '      "a": 1',
        "      }",
        "      error: boom",
        "[tool] next",
    ]
    assert_eq(_consume_continuation(lines, 0), 3, "stops before error:")
    assert_eq(_consume_continuation(lines, 3), 4, "stops before [tool]")


def test_unfold_joins_wrapped_block():
    block = ["      args: {", '      "a": 1', "      }"]
    assert_eq(_unfold(block), 'args: { "a": 1 }', "unfold")


if __name__ == "__main__":
    failures = 0
    count = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            count += 1
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{count - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)