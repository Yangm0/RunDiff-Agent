# RunDiff Agent

**Evidence-grounded investigation of machine-learning experiment runs.**

RunDiff Agent compares two local ML run folders, lets an LLM choose additional read-only investigations, and produces a report whose observations cite verifiable source evidence. It is a compact example of tool-using agent engineering for experiment debugging.

The repository contains only synthetic demonstration runs. The HTTP API accepts bundled case IDs and cannot read arbitrary server paths.

## What it does

- Validates manifest.json, config.json, and metrics.json with Pydantic v2.
- Compares nested configuration, runtime and dependency metadata, and explicitly defined metrics.
- Reads bounded, line-numbered log excerpts without executing or obeying log contents.
- Runs a LangGraph StateGraph with real tool calling, per-request state, duplicate-call checks, and call/round budgets.
- Supports a clearly labeled deterministic Mock Mode without credentials and an OpenAI-compatible Live Mode.
- Verifies evidence IDs, run ownership, JSON paths and values, log line numbers, and excerpts before building a report.
- Writes report.json, report.md, and trace.jsonl.
- Provides a shared application service behind a Typer CLI and FastAPI API.
- Includes eight synthetic cases with answer keys stored outside the agent input folders.

## Architecture

    Two run directories
        -> Pydantic input validation
        -> Initial deterministic comparisons
        -> Bounded baseline summary passed to the agent as untrusted data
        -> LangGraph agent decision
        -> Validated read-only tool dispatcher
        -> Agent observation and next decision
        -> Evidence validation and structured report
        -> JSON, Markdown, and JSONL trace

The CLI and FastAPI handlers call the same ApplicationService.

## Requirements and install

Python 3.11 or newer is required. The project was built with Python 3.13 on Windows.

    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -e ".[dev]"

If PowerShell blocks activation, invoke .\.venv\Scripts\python.exe and .\.venv\Scripts\rundiff.exe directly.

## Demo without an API key

    rundiff analyze --example learning-rate-wave --mode mock --output-dir outputs\demo
    rundiff compare examples\cases\learning-rate-wave\run_a examples\cases\learning-rate-wave\run_b
    rundiff eval --mode mock --output-dir outputs\evaluation

Mock Mode executes the same LangGraph route and deterministic tools, using a scripted stand-in for predictable local validation. It is not an LLM response or a live model evaluation score.

## Live LLM mode

Set credentials in the process environment, or use a local .env file with your own loader. The .env file is ignored by Git.

    $env:OPENAI_API_KEY = "your-key"
    $env:OPENAI_MODEL = "gpt-4o-mini"
    $env:OPENAI_BASE_URL = "https://your-provider.example/v1"
    rundiff analyze --example learning-rate-wave --mode live --output-dir outputs\live

OPENAI_BASE_URL is optional. Live Mode requires an actual API call. Missing credentials produce an explicit error; the program does not label Mock Mode as a live investigation. Provider calls incur normal usage charges. Token counts are recorded only when returned by the provider.

## CLI

- rundiff compare RUN_A RUN_B prints deterministic configuration, environment, and metric comparisons.
- rundiff analyze --example CASE_ID --mode mock|live runs the agent and writes the three report artifacts.
- rundiff analyze --run-a PATH --run-b PATH --mode mock|live analyzes two local folders.
- rundiff eval --mode mock|live evaluates the bundled synthetic cases. Answer keys are loaded only after analysis finishes.

Use rundiff --help or rundiff analyze --help for options.

## FastAPI

Start the API with:

    uvicorn rundiff.api:app --reload

Open the API docs at http://127.0.0.1:8000/docs.

    GET /health
    GET /api/examples
    POST /api/analyze
    Content-Type: application/json
    {"example_id":"learning-rate-wave","mode":"mock"}

The API intentionally accepts only a validated bundled example ID. It does not accept a path from the request.

## Input protocol

Each run directory has three required JSON files and an optional train.log:

- manifest.json: unique run_id, source commit, dataset, Python/framework versions, dependency versions, and optional platform.
- config.json: any JSON object; nested fields are compared recursively.
- metrics.json: an object containing metrics; each metric declares a finite final value, direction (maximize, minimize, or neutral), and explicit unit. Optional history is a finite numeric sequence.
- train.log: UTF-8 plain text. Evidence refers to exact one-based line numbers. Missing logs are reported.

NaN and Infinity are rejected. Unknown metrics are compared numerically only when both sides declare the same direction and unit. A metric name is never used to infer its unit or optimization direction.

## Example report output

For the bundled learning-rate-wave case, a real Mock Mode run reported:

    status: complete
    deterministic findings: optimizer.learning_rate 0.001 -> 0.003;
      validation_accuracy 0.91 -> 0.89; validation_loss 0.22 -> 0.29
    agent tool calls: 2
    decision rounds: 3
    evidence records: 9; validation: valid

The demo command writes report.json, report.md, and trace.jsonl under outputs/demo. The JSON includes evidence records and validation status; the JSONL trace includes model and tool steps, outcome, duration, and evidence count.

## Evaluation and verification

rundiff eval --mode mock reports per-case planted-change recall, key-difference precision, evidence validity, unsupported claim counts, tool success rate, steps, and elapsed time. Its label is mock_flow_validation. If a live API key is configured, the runner can also record a one-shot live baseline; it does not invent a score when calls fail.

The retained CI workflow installs development dependencies, runs Ruff, and runs pytest. Its automatic `push` and `pull_request` triggers are temporarily paused for this release; only manual dispatch is enabled while the GitHub Actions quota is exhausted. After the quota resets, restore the `push` and `pull_request` entries in `.github/workflows/ci.yml`.

Test results and evaluation results in this README describe only commands actually executed for the current revision. GitHub Actions was not run for this release; the checks below were performed locally.

Verified locally with Python 3.13.2:

- Ruff: ruff check . — passed.
- Tests: pytest -q — 51 passed.
- CLI: the learning-rate-wave Mock Mode analysis completed and wrote all three output files.
- CLI compare: the learning-rate-wave run folders were compared locally.
- Evaluation: 8/8 synthetic cases; mean planted-change recall 1.0, mean key-difference precision 1.0, and Mock tool-call success rate 1.0.
- FastAPI: TestClient passed health, example listing, Swagger, successful analysis, arbitrary-path rejection, and the missing-key Live Mode response.
- No OPENAI_API_KEY was configured. Live LLM calls and the one-shot live baseline were not run. The evaluation scores above are deterministic and Mock workflow checks, not LLM quality results.

## Security and limits

- The agent is read-only: tools do not execute shell commands or model-generated code.
- Run contents are untrusted input; prompt-like text is evidence data, not an instruction.
- Log inspection is bounded by file size, line range, result count, and excerpt size.
- Sensitive-looking credential values are redacted from evidence and report artifacts.
- The first version is a local research/debugging tool. It has no database, job queue, frontend, MCP server, or Docker deployment.
- Evidence can establish that values differ and that a log line exists; it cannot establish that one difference caused another.

## Positioning

RunDiff Agent is a small evidence and agent-workflow demo. It is not a replacement for experiment trackers such as MLflow or Weights & Biases, which provide persistent run management, artifact storage, and broader team workflows. It focuses on investigating a pair of exported runs and making report claims traceable to source files.

## 中文简介

RunDiff Agent 面向机器学习实验排查：比较两个本地运行目录，由 LangGraph Agent 选择只读调查工具，并为报告中的客观发现附上可核验的配置、指标或日志证据。Mock 模式用于验证工作流，不代表真实大模型评测结果。

