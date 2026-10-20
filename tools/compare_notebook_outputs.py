"""Fail if a fresh run of the notebook prints different results than the committed one.

Usage: python tools/compare_notebook_outputs.py COMMITTED.ipynb FRESH.ipynb

Compares what every code cell printed to stdout. Timings, stderr (library
warnings) and rich outputs (plots, progress widgets) are ignored.
"""
import difflib
import json
import re
import sys

TIMING = re.compile(r"(completed in|calculated in) [\d.]+s")


def printed_output(path):
    with open(path) as f:
        cells = json.load(f)["cells"]
    return [
        TIMING.sub(r"\1 <time>", "".join(
            "".join(o.get("text", "")) for o in cell.get("outputs", [])
            if o.get("output_type") == "stream" and o.get("name") == "stdout"))
        for cell in cells if cell["cell_type"] == "code"
    ]


def differences(committed, fresh):
    old, new = printed_output(committed), printed_output(fresh)
    if len(old) != len(new):
        return [f"code cell count differs: {len(old)} committed vs {len(new)} fresh"]
    return [
        f"code cell {i}:\n" + "\n".join(difflib.unified_diff(
            a.splitlines(), b.splitlines(), "committed", "fresh", lineterm="", n=1))
        for i, (a, b) in enumerate(zip(old, new)) if a != b
    ]


if __name__ == "__main__":
    found = differences(sys.argv[1], sys.argv[2])
    print("\n\n".join(found) if found else "Printed output matches the committed notebook.")
    sys.exit(1 if found else 0)
