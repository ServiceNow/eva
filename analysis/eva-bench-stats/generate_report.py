"""HTML report generator for EVA-Bench statistics.

Run from project root:
    uv run python analysis/eva-bench-stats/generate_report.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "output_processed" / "eva-bench-stats"


def build_report(output_dir: Path | None = None, inline_js: bool = False) -> Path:
    """Build a single HTML report with all analyses.

    Args:
        output_dir: Directory to write the report. Defaults to
                    output_processed/eva-bench-stats/reports/
        inline_js: Embed Plotly JS inline (fully self-contained).
                   Default (False) uses CDN — lighter file.

    Returns:
        Path to the generated HTML file.
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_plotlyjs = "inline" if inline_js else "cdn"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"report_{timestamp}.html"

    sections = _build_sections(include_plotlyjs=include_plotlyjs)
    html = _assemble_html(sections, timestamp=timestamp)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_sections(include_plotlyjs: str) -> list[str]:
    return [
        _perturbations_section(include_plotlyjs=include_plotlyjs),
        _placeholder_section("CIs", "Confidence Intervals"),
        _placeholder_section("variance", "Variance Analysis"),
        _placeholder_section("frontier", "Frontier Analysis"),
    ]


def _perturbations_section(include_plotlyjs: str) -> str:
    return "<section><h2>Perturbation Tests</h2><p>Analysis not yet implemented.</p></section>"


def _placeholder_section(area: str, title: str) -> str:
    return f"<section><h2>{title}</h2><p>Coming soon.</p></section>"


def _assemble_html(sections: list[str], timestamp: str) -> str:
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>EVA-Bench Statistics — {timestamp}</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 2em; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5em; }}
    section {{ margin-bottom: 3em; }}
    h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }}
  </style>
</head>
<body>
  <h1>EVA-Bench Statistics — {timestamp}</h1>
  {body}
</body>
</html>"""


if __name__ == "__main__":
    path = build_report()
    print(f"Report saved to {path}")
