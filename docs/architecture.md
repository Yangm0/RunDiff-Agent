# Architecture and trust boundaries

The CLI, FastAPI handlers, and evaluator call ApplicationService. Run loading, deterministic tools, report generation, and the LangGraph coordinator remain separate modules.

Each request creates a RunPair, a LangGraph state, and (in Live Mode) its model client. There is no shared mutable run cache. The initial-comparison node records deterministic config, environment, and metric results, then passes a bounded summary and evidence IDs to the model as untrusted data. The agent can issue validated calls to compare_configs, compare_environments, analyze_metrics, and inspect_logs. A bounded dispatcher converts exceptions into tool observations; call budgets, decision rounds, and duplicate calls are tracked in the state.

Tool outputs contain stable IDs based on source identity and value. The validator confirms that a citation belongs to one of the two input run IDs and that its protocol filename, JSON field/value, or line number and excerpt resolve against that run. Unsupported citation IDs are removed and the affected claim becomes UNVERIFIED. Numeric observations in the report are assembled from deterministic results rather than copied from model prose.

## Trust boundaries

Run contents and model responses are untrusted. The system prompt tells the LLM that tool output is data, and tools have no shell or write capability. Local CLI input paths are provided by the operator; the HTTP API does not accept paths and resolves only bundled example IDs. Inputs have size and schema constraints. Secret-like text is redacted from report evidence and trace fields.

The static tools establish observed differences, not causal explanations. Mock mode deliberately uses a deterministic stand-in so the graph can be tested without credentials. Mock results are labeled and should not be compared as LLM quality measurements.

