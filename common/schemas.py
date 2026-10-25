from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

Label = Literal["malicious", "benign"]
Category = Literal[
    "reward_hacking",        # gaming tests/metrics instead of solving the task
    "sandbagging",           # deliberately underperforming
    "user_deception",        # says one thing in CoT, tells user another
    "permission_seeking",    # quietly expanding its own access/resources
    "oversight_subversion",  # disabling logging, monitors, or checks
    "hint_misuse",           # uses a leaked answer/hint and hides it
    "none",
]
Difficulty = Literal["obvious", "subtle", "hard_negative", "clean"]
Source = Literal["synthetic", "real", "evasion_r1", "evasion_r2", "fixture"]

@dataclass
class Trace:
    id: str
    task: str                 # the task/prompt the model was given
    cot: str                  # the chain of thought
    final_answer: str         # what the model told the user / did
    label: Label
    category: Category        # "none" iff label == "benign"
    difficulty: Difficulty
    source: Source
    parent_id: Optional[str] = None   # evasion rewrites point to the original

@dataclass
class Stage1Result:
    flag: bool
    category: Category
    confidence: float         # 0..1
    suspicious_span: str      # exact quote from the CoT, "" if none

@dataclass
class Stage2Result:
    confirmed: bool           # False = stage 2 overturned the flag
    explanation: str          # plain English, 2–4 sentences
    predicted_next_action: str

@dataclass
class TokenUsage:
    stage1_in: int = 0
    stage1_out: int = 0
    stage2_in: int = 0
    stage2_out: int = 0

@dataclass
class MonitorResult:
    trace_id: str
    stage1: Stage1Result
    stage2: Optional[Stage2Result]    # None when stage 1 did not flag
    final_flag: bool                  # stage1.flag and (stage2 is None or stage2.confirmed)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0

def to_dict(obj):  # use for JSONL writing
    return asdict(obj)
