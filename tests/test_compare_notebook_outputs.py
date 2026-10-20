import json

from tools.compare_notebook_outputs import differences


def write_notebook(path, cell_texts, stderr=""):
    cells = [{"cell_type": "markdown", "source": ["# title"], "metadata": {}}]
    for text in cell_texts:
        outputs = [{"output_type": "stream", "name": "stdout", "text": [text]}]
        if stderr:
            outputs.append({"output_type": "stream", "name": "stderr", "text": [stderr]})
        outputs.append({"output_type": "display_data", "data": {"text/plain": ["<widget>"]}, "metadata": {}})
        cells.append({"cell_type": "code", "source": [], "metadata": {}, "outputs": outputs, "execution_count": 1})
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    return path


def test_identical_output_passes(tmp_path):
    a = write_notebook(tmp_path / "a.ipynb", ["Qini= +5.87\n", "AUC 0.8139\n"])
    b = write_notebook(tmp_path / "b.ipynb", ["Qini= +5.87\n", "AUC 0.8139\n"])

    assert differences(a, b) == []


def test_timings_stderr_and_widgets_are_ignored(tmp_path):
    a = write_notebook(tmp_path / "a.ipynb", ["Training completed in 7.5s\nSHAP values calculated in 0.0s\n"])
    b = write_notebook(tmp_path / "b.ipynb", ["Training completed in 12.1s\nSHAP values calculated in 3.4s\n"],
                       stderr="Failed to import duecredit\n")

    assert differences(a, b) == []


def test_changed_result_is_reported_with_its_cell(tmp_path):
    a = write_notebook(tmp_path / "a.ipynb", ["same\n", "Qini= +5.87\n"])
    b = write_notebook(tmp_path / "b.ipynb", ["same\n", "Qini= +5.88\n"])

    [report] = differences(a, b)
    assert report.startswith("code cell 1") and "-Qini= +5.87" in report and "+Qini= +5.88" in report


def test_different_cell_count_is_reported(tmp_path):
    a = write_notebook(tmp_path / "a.ipynb", ["x\n"])
    b = write_notebook(tmp_path / "b.ipynb", ["x\n", "y\n"])

    assert differences(a, b) == ["code cell count differs: 1 committed vs 2 fresh"]


def test_last_digit_float_noise_is_tolerated(tmp_path):
    # float32 predictions can differ in the last bit between macOS and Linux
    a = write_notebook(tmp_path / "a.ipynb", ["353  0.681224  1049.958296  Persuadable\n"])
    b = write_notebook(tmp_path / "b.ipynb", ["353  0.681224  1049.958205  Persuadable\n"])

    assert differences(a, b) == []


def test_changed_text_with_equal_numbers_is_reported(tmp_path):
    a = write_notebook(tmp_path / "a.ipynb", ["Selected by cross-validation: X-Learner\n"])
    b = write_notebook(tmp_path / "b.ipynb", ["Selected by cross-validation: S-Learner\n"])

    assert len(differences(a, b)) == 1
