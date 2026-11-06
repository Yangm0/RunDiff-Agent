# RunDiff Agent

RunDiff Agent investigates two machine-learning experiment runs. It compares their configuration, environment, metrics, and logs, then uses a LangGraph agent to select additional read-only investigations. Report findings cite source evidence that the application checks against the input runs.

The HTTP API works with bundled examples only. The CLI can also analyze two local run directories.

## Features

- Pydantic validation for run manifests, configuration, and metrics.
- Deterministic comparisons for configuration, environment, and metric history.
- LangGraph tool calling with bounded investigation and read-only tools.
- Evidence checks for run ownership, source fields, metric values, and log excerpts.
- JSON, Markdown, and JSONL report output through a Typer CLI and FastAPI.
- Mock mode for a local demonstration without model credentials; Live mode for an OpenAI-compatible endpoint.

## Install

Python 3.11 or newer is required.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install .
```

## Run a demo

```powershell
rundiff analyze --example learning-rate-wave --mode mock --output-dir outputs\demo
rundiff compare examples\cases\learning-rate-wave\run_a examples\cases\learning-rate-wave\run_b
```

Mock mode uses a deterministic stand-in and does not make a live model request.

## Live mode

Set credentials in the process environment. `OPENAI_BASE_URL` is optional when using the default OpenAI endpoint.

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "your-model"
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
rundiff analyze --example learning-rate-wave --mode live --output-dir outputs\live
```

Provider calls may incur charges. Token usage is recorded when the provider returns it. OpenAI-compatible providers can differ in tool-calling and structured-output support.

## FastAPI

```powershell
uvicorn rundiff.api:app --reload
```

The API exposes `/health`, `/api/examples`, and `/api/analyze`. It accepts bundled example IDs rather than filesystem paths.

## Input format

Each run directory contains `manifest.json`, `config.json`, and `metrics.json`, with an optional UTF-8 `train.log`. Metrics declare their values, units, and optimization directions explicitly. The application rejects invalid or non-finite input values.

Run contents and model responses are treated as untrusted data. Static comparisons show differences but do not establish causation. The tools are read-only and do not execute run contents or model-generated code.
