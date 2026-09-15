"""
context.py — shared scratchpad for one workflow run.

Tools write their structured results here as they execute, so the orchestrator
can (a) hand exact data from Agent 1 to Agent 2 and (b) include the precise
numbers — not just the LLM's prose — in the final output. One instance is
created per `run_workflow` call and threaded through both agents' tools.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WorkflowContext:
    profile: Optional[Dict] = None          # demographic profile from the Intake API
    nutrition_report: Optional[Dict] = None  # Agent 1's structured gap report
    features: Optional[Dict] = None          # the 106-column model input that was sent
    prediction: Optional[Dict] = None        # Agent 2's model result {label, probability, raw}
