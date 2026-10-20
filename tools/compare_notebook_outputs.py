"""Fail if a fresh run of the notebook prints different results than the committed one.

Usage: python tools/compare_notebook_outputs.py COMMITTED.ipynb FRESH.ipynb

Compares what every code cell printed to stdout. Timings, stderr (library
warnings) and rich outputs (plots, progress widgets) are ignored. Numbers may
differ by a relative 1e-6: XGBoost's float32 predictions can differ in the last
bit between macOS and Linux.
"""
import difflib
import json
import math
import re
import sys

TIMING = re.compile(r"(completed in|calculated in) [\d.]+s")
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def printed_output(path):
    with open(path) as f:
        cells = json.load(f)["cells"]
    return [
        TIMING.sub(r"\1 <time>", "".join(
            "".join(o.get("text", "")) for o in cell.get("outputs", [])
            if o.get("output_type") == "stream" and o.get("name") == "stdout"))
        for cell in cells if cell["cell_type"] == "code"
    ]


def same_line(a, b, rel_tol=1e-6):
    """Equal text, and every number equal up to `rel_tol`."""
    if NUMBER.sub("#", a) != NUMBER.sub("#", b):
        return False
    return all(math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=1e-9)
               for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)))


def same_output(a, b):
    a, b = a.splitlines(), b.splitlines()
    return len(a) == len(b) and all(same_line(x, y) for x, y in zip(a, b))


def differences(committed, fresh):
    old, new = printed_output(committed), printed_output(fresh)
    if len(old) != len(new):
        return [f"code cell count differs: {len(old)} committed vs {len(new)} fresh"]
    return [
        f"code cell {i}:\n" + "\n".join(difflib.unified_diff(
            a.splitlines(), b.splitlines(), "committed", "fresh", lineterm="", n=1))
        for i, (a, b) in enumerate(zip(old, new)) if not same_output(a, b)
    ]


if __name__ == "__main__":
    found = differences(sys.argv[1], sys.argv[2])
    print("\n\n".join(found) if found else "Printed output matches the committed notebook.")
    sys.exit(1 if found else 0)
