"""Execute notebook cells and attach outputs and figures to notebooks/01_eda_and_preprocessing.ipynb."""

from __future__ import annotations

import base64
import contextlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from steam_player_forecasting.data.loader import get_project_root


def execute_and_populate_notebook() -> None:
    root = get_project_root()
    notebook_path = root / "notebooks" / "01_eda_and_preprocessing.ipynb"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    exec_globals: dict = {}
    exec_count = 1

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        code = "".join(cell["source"])
        outputs = []
        stdout_buf = io.StringIO()

        # Intercept plt.show to capture matplotlib figures
        captured_figs = []

        def custom_show(*args, **kwargs):
            fig = plt.gcf()
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                captured_figs.append(img_b64)
            plt.close(fig)

        def custom_display(obj):
            if isinstance(obj, pd.DataFrame):
                stdout_buf.write(obj.to_string() + "\n")
            else:
                stdout_buf.write(str(obj) + "\n")

        exec_globals["plt"] = plt
        exec_globals["display"] = custom_display
        orig_show = plt.show
        plt.show = custom_show

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
                exec(code, exec_globals)
        finally:
            plt.show = orig_show

        # Add text output if any
        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            outputs.append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in stdout_text.rstrip("\n").split("\n")],
                }
            )

        # Add image outputs if any
        for b64_img in captured_figs:
            outputs.append(
                {
                    "data": {
                        "image/png": b64_img,
                        "text/plain": ["<Figure size ...>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                }
            )

        cell["execution_count"] = exec_count
        cell["outputs"] = outputs
        exec_count += 1

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Successfully executed notebook and saved all outputs!")


if __name__ == "__main__":
    execute_and_populate_notebook()
