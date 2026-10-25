"""Typer commands for local comparison, analysis, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .agent import LiveModelUnavailable
from .evaluation import run_evaluation
from .loader import RunInputError
from .service import ApplicationService

app = typer.Typer(
    name="rundiff",
    help="Evidence-grounded comparison and investigation of ML experiment runs.",
    no_args_is_help=True,
)
service = ApplicationService()


@app.command()
def compare(
    run_a: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    run_b: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
) -> None:
    """Print deterministic configuration, environment, and metric comparisons."""
    try:
        typer.echo(
            json.dumps(service.compare(run_a, run_b), ensure_ascii=False, indent=2, default=str)
        )
    except RunInputError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def analyze(
    example: Annotated[
        str | None, typer.Option("--example", help="Bundled synthetic case ID.")
    ] = None,
    run_a: Annotated[Path | None, typer.Option("--run-a", exists=True, file_okay=False)] = None,
    run_b: Annotated[Path | None, typer.Option("--run-b", exists=True, file_okay=False)] = None,
    mode: Annotated[str, typer.Option("--mode", help="Agent mode: mock or live.")] = "mock",
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Directory for report and trace.")
    ] = Path("outputs/latest"),
) -> None:
    """Investigate one approved example or a pair of local run directories."""
    if mode not in {"mock", "live"}:
        raise typer.BadParameter("mode must be mock or live")
    if example and (run_a or run_b):
        raise typer.BadParameter("Choose --example or --run-a/--run-b, not both.")
    if bool(run_a) != bool(run_b) or (not example and not run_a):
        raise typer.BadParameter("Provide --example or both --run-a and --run-b.")
    try:
        if example:
            report = service.analyze_example(example, mode=mode, output_dir=output_dir)
        else:
            assert run_a and run_b
            report = service.analyze(run_a, run_b, mode=mode, output_dir=output_dir)
        typer.echo(
            f"{report['status']} ({report['mode']}): {report['summary']}\n"
            f"Report files written to {output_dir.resolve()}"
        )
    except (RunInputError, LiveModelUnavailable) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def eval(
    mode: Annotated[
        str, typer.Option("--mode", help="Run mock flow validation or live agent evaluation.")
    ] = "mock",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("outputs/evaluation"),
) -> None:
    """Evaluate the synthetic cases; answer keys are never provided to the agent."""
    if mode not in {"mock", "live"}:
        raise typer.BadParameter("mode must be mock or live")
    try:
        result = run_evaluation(mode, output_dir)
        typer.echo(
            f"{result['evaluation_mode']}: {result['case_count']} cases; "
            f"results written to {output_dir.resolve()}"
        )
    except (RunInputError, LiveModelUnavailable) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
