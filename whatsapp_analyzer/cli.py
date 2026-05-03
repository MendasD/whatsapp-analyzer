"""
Click CLI for whatsapp-analyser.

Commands:
    analyze  Run the full pipeline on one group, print a summary table.
    compare  Compare several groups side by side.
    serve    Launch the Streamlit web interface.

All heavy imports are lazy (inside command bodies).
No raw traceback ever reaches the user.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli() -> None:
    """WhatsApp conversation analyser."""


@cli.command()
@click.option(
    "--input", "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to chat.zip, _chat.txt, or folder.",
)
@click.option("--topics", default=5, show_default=True, help="Number of LDA topics.")
@click.option("--output", "output_dir", default="reports", show_default=True,
              help="Directory for report.html.")
def analyze(input_path: str, topics: int, output_dir: str) -> None:
    """Run the full analysis pipeline on a single group."""
    try:
        from whatsapp_analyzer.core import WhatsAppAnalyzer

        console.print(f"[bold]Analysing[/bold] {input_path} …")
        az = WhatsAppAnalyzer(input_path, n_topics=topics, output_dir=output_dir)
        results = az.run()

        df = results["df_clean"]
        period_start = df["timestamp"].min().strftime("%Y-%m-%d")
        period_end = df["timestamp"].max().strftime("%Y-%m-%d")

        table = Table(title=f"Analysis — {results['group_name']}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Group", results["group_name"])
        table.add_row("Messages", str(len(df)))
        table.add_row("Participants", str(int(df["author"].nunique())))
        table.add_row("Period", f"{period_start} → {period_end}")

        topics_result = results.get("topics")
        if topics_result is not None:
            top = topics_result["group_topics"].sort_values("weight", ascending=False)
            for _, row in top.head(3).iterrows():
                table.add_row("Top topic", row["topic_label"])

        console.print(table)
        console.print(f"[green]Report written to[/green] {results['report_path']}")

    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.option(
    "--input", "input_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Paths to group chat files (repeat for each group).",
)
@click.option("--output", "output_dir", default="reports", show_default=True,
              help="Directory for comparison_report.html.")
def compare(input_paths: tuple[str, ...], output_dir: str) -> None:
    """Compare multiple WhatsApp groups side by side."""
    try:
        from whatsapp_analyzer.core import WhatsAppAnalyzer
        from whatsapp_analyzer.comparator import GroupComparator

        analyzers = []
        for p in input_paths:
            console.print(f"[bold]Loading[/bold] {p} …")
            az = WhatsAppAnalyzer(p, output_dir=output_dir)
            az.run()
            analyzers.append(az)

        cmp = GroupComparator(analyzers)
        activity = cmp.compare_activity()

        table = Table(title="Group Comparison — Activity")
        table.add_column("Group", style="cyan")
        for col in activity.columns:
            table.add_column(str(col), style="white")
        for group_name, row in activity.iterrows():
            table.add_row(str(group_name), *[str(v) for v in row])

        console.print(table)

        report_path = cmp.report(Path(output_dir))
        console.print(f"[green]Comparison report written to[/green] {report_path}")

    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@cli.command()
def serve() -> None:
    """Launch the Streamlit web interface."""
    import subprocess

    app_path = Path(__file__).parent / "app.py"
    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
