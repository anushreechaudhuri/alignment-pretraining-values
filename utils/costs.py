"""
Real-time cost tracking for API calls across the extraction pipeline.

Monitors spending on OpenAI and Anthropic APIs, logs each call's token usage
and cost, and raises warnings if spending exceeds budget thresholds. All costs
are saved to a persistent log file so spending can be audited after runs.

Pricing as of March 2026 (update if prices change):
- GPT-5.2: $1.75/$14.00 per MTok (batch: $0.875/$7.00)
- Claude Sonnet 4.6: $3.00/$15.00 per MTok (batch: $1.50/$7.50)
- Claude Opus 4.6: $5.00/$25.00 per MTok (batch: $2.50/$12.50)
"""

import json
import time
from pathlib import Path
from datetime import datetime


# Pricing per million tokens (input, output) for standard and batch
PRICING = {
    "gpt-5.2": {"input": 1.75, "output": 14.00, "batch_input": 0.875, "batch_output": 7.00},
    "gpt-5": {"input": 1.25, "output": 10.00, "batch_input": 0.625, "batch_output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "batch_input": 0.075, "batch_output": 0.30},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00, "batch_input": 2.50, "batch_output": 12.50},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "batch_input": 1.50, "batch_output": 7.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "batch_input": 0.50, "batch_output": 2.50},
}

# Default budget cap in USD - pipeline will warn (not stop) if exceeded
DEFAULT_BUDGET_CAP = 50.00


class CostTracker:
    """
    Tracks cumulative API costs across a pipeline run.

    This class is the central point for monitoring how much money each pipeline
    step spends on language-model inference. Every API call should be recorded
    through ``log_call``, which computes the dollar cost from the model's
    per-token pricing, updates running totals, and appends a structured JSON
    line to a persistent log file. If cumulative spending crosses a configurable
    budget cap, a warning is printed to standard output so operators can decide
    whether to continue.

    The cost log (a JSONL file) is designed for post-hoc auditing: each line
    records the model, token counts, per-call cost, cumulative cost, and any
    caller-supplied metadata (e.g., which prompt template or document was being
    processed). This makes it straightforward to attribute costs to specific
    pipeline stages or experimental conditions.

    Usage:
        tracker = CostTracker(budget_cap=30.0)
        tracker.log_call("gpt-5.2", input_tokens=1500, output_tokens=400)
        tracker.log_call("claude-sonnet-4-6", input_tokens=1500, output_tokens=400)
        print(tracker.summary())

    Costs are logged to outputs/data/cost_log.jsonl for post-hoc auditing.
    """

    def __init__(self, budget_cap=DEFAULT_BUDGET_CAP, log_path=None, is_batch=False):
        """
        Initialize a new cost tracker for a pipeline run.

        Args:
            budget_cap: Maximum expected spend in USD. The tracker prints
                warnings to standard output if cumulative cost exceeds this
                value, but it does not halt execution. Set this to the amount
                you have budgeted for the run so surprises surface early.
            log_path: Path to the JSONL file where per-call cost records are
                appended. Defaults to ``outputs/data/cost_log.jsonl`` relative
                to the project root. Parent directories are created
                automatically if they do not exist.
            is_batch: If True, use batch pricing (typically 50 percent cheaper)
                instead of standard on-demand pricing. Enable this when
                submitting requests through the provider's batch API.
        """
        self.budget_cap = budget_cap
        self.is_batch = is_batch
        self.total_cost = 0.0
        self.calls_by_model = {}
        self.cost_by_model = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if log_path is None:
            log_path = Path(__file__).parent.parent / "outputs" / "data" / "cost_log.jsonl"
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_pricing(self, model):
        """
        Look up per-token pricing for a model.

        Performs an exact match against the ``PRICING`` dictionary first. If no
        exact match is found, falls back to a substring match (useful when
        callers pass versioned model identifiers like ``gpt-5.2-2026-03``).
        If neither match succeeds, GPT-5.2 pricing is used as a conservative
        fallback and a warning is printed.

        Args:
            model: The model identifier string returned by the API or supplied
                by the caller.

        Returns:
            dict: A pricing dictionary with keys ``input``, ``output``,
            ``batch_input``, and ``batch_output``, each giving the cost in
            USD per million tokens.
        """
        if model not in PRICING:
            for key in PRICING:
                if key in model or model in key:
                    return PRICING[key]
            print(f"  WARNING: Unknown model '{model}', using GPT-5.2 pricing as fallback")
            return PRICING["gpt-5.2"]
        return PRICING[model]

    def log_call(self, model, input_tokens, output_tokens, metadata=None):
        """
        Record a single API call's cost.

        Computes the dollar cost of the call from the model's pricing and the
        number of tokens consumed, updates all internal counters, writes a
        structured log line, and checks budget thresholds.

        Args:
            model: Model identifier (e.g., ``"gpt-5.2"``,
                ``"claude-sonnet-4-6"``). Must match or partially match a key
                in the module-level ``PRICING`` dictionary.
            input_tokens: Number of input (prompt) tokens billed by the API.
            output_tokens: Number of output (completion) tokens billed by the
                API.
            metadata: Optional dictionary with extra context for the log entry.
                Useful for recording which prompt template, document, or
                pipeline stage generated this call so costs can be sliced
                during auditing.

        Returns:
            float: Cost of this individual call in USD.
        """
        pricing = self._get_pricing(model)

        if self.is_batch:
            input_cost = (input_tokens / 1_000_000) * pricing["batch_input"]
            output_cost = (output_tokens / 1_000_000) * pricing["batch_output"]
        else:
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]

        call_cost = input_cost + output_cost
        self.total_cost += call_cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Track per-model
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
        self.cost_by_model[model] = self.cost_by_model.get(model, 0) + call_cost

        # Write to log file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "call_cost_usd": round(call_cost, 6),
            "cumulative_cost_usd": round(self.total_cost, 4),
            "budget_remaining_usd": round(self.budget_cap - self.total_cost, 4),
        }
        if metadata:
            log_entry["metadata"] = metadata

        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Budget warnings
        if self.total_cost > self.budget_cap:
            print(f"  BUDGET WARNING: ${self.total_cost:.2f} spent, cap is ${self.budget_cap:.2f}")
        elif self.total_cost > self.budget_cap * 0.8:
            print(f"  Budget alert: ${self.total_cost:.2f} / ${self.budget_cap:.2f} (80%+ used)")

        return call_cost

    def summary(self):
        """
        Return a human-readable summary of all costs recorded so far.

        The summary includes total spend, budget utilization, aggregate token
        counts, and a per-model breakdown of call counts and costs. This is
        intended for printing at the end of a pipeline run or checkpoint.

        Returns:
            str: A multi-line formatted string suitable for printing to the
            console or writing to a report file.
        """
        lines = [
            f"Cost Summary",
            f"{'='*40}",
            f"Total cost: ${self.total_cost:.4f}",
            f"Budget cap: ${self.budget_cap:.2f}",
            f"Budget remaining: ${self.budget_cap - self.total_cost:.2f}",
            f"Total input tokens: {self.total_input_tokens:,}",
            f"Total output tokens: {self.total_output_tokens:,}",
            f"",
            f"By model:",
        ]
        for model in sorted(self.cost_by_model.keys()):
            n_calls = self.calls_by_model[model]
            cost = self.cost_by_model[model]
            lines.append(f"  {model}: {n_calls} calls, ${cost:.4f}")

        return "\n".join(lines)

    def is_over_budget(self):
        """
        Check whether cumulative spending has exceeded the budget cap.

        Returns:
            bool: True if total cost is strictly greater than the budget cap.
        """
        return self.total_cost > self.budget_cap

    def get_total_cost(self):
        """
        Return the total cost accumulated across all recorded API calls.

        Returns:
            float: Cumulative cost in USD.
        """
        return self.total_cost
