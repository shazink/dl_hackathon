"""Small Plotly constructors used by the Streamlit dashboard."""

from __future__ import annotations

import plotly.express as px
import pandas as pd


def comparison_chart(rows: list[dict], y: str, title: str):
    return px.bar(pd.DataFrame(rows), x="method", y=y, color="method", title=title, template="plotly_dark")


def matrix_heatmap(matrix: list[list[float]], rows: list[str], columns: list[str], title: str):
    return px.imshow(matrix, x=columns, y=rows, text_auto=".3f", aspect="auto", title=title, template="plotly_dark")


def scatter_tradeoff(rows: list[dict]):
    return px.scatter(
        pd.DataFrame(rows), x="balanced_accuracy_forgetting", y="validation_balanced_accuracy",
        color="method", hover_name="method", title="Validation performance vs forgetting", template="plotly_dark",
    )
