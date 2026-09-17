from __future__ import annotations


def slugify_plot_name(text: str) -> str:
    """
    Convert a plot label or filename component into a simple filesystem-safe slug.
    """
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("@", "at")
    )