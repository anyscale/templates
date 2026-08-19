"""
Single-turn environment for NCT-CRC-HE tissue-type classification.

The model sees one H&E patch plus the 9-class list, reasons, and commits to a
final answer inside <answer>...</answer> tags. One step, then done -- no tool
protocol, no turn loop. This is the PathAI shape: single-turn diagnosis with a
rule reward now, a reward *model* later (Phase 3).

Modeled on examples/train/geometry3k/env.py with the calc_score tool machinery
(TOOL_CALL_RE, SUPPORTED_TOOL_NAMES, _extract_tool_call, _build_tool_feedback,
the multi-turn branches in step()) stripped out.

Reward (additive, matching context/grpo-step-anatomy.md Stage 3):
    1.0  answer matches ground truth
  + 0.2  well-formed <answer>...</answer> tags, awarded even when wrong
  = 1.2  correct and well-formed
    0.0  no parseable answer tags
The format bonus is what keeps a GRPO group from collapsing to all-zero
advantages early in training, when the model gets the class wrong but is
learning the output contract.
"""

import re
from typing import Any, Dict, List

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

CORRECT_REWARD = 1.0
FORMAT_BONUS = 0.2

# Canonical class names, as emitted by nct_crc_dataset.py into reward_spec.
CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]

# Accepted spellings -> canonical name. The dataset's own abbreviations plus the
# obvious shorthands, so a correct diagnosis is not scored wrong on phrasing.
ALIASES = {
    "adi": "adipose",
    "adipose tissue": "adipose",
    "fat": "adipose",
    "back": "background",
    "deb": "debris",
    "lym": "lymphocytes",
    "lymphocyte": "lymphocytes",
    "muc": "mucus",
    "mucin": "mucus",
    "mus": "smooth muscle",
    "muscle": "smooth muscle",
    "norm": "normal colon mucosa",
    "normal mucosa": "normal colon mucosa",
    "normal colon": "normal colon mucosa",
    "str": "cancer-associated stroma",
    "stroma": "cancer-associated stroma",
    "cancer associated stroma": "cancer-associated stroma",
    "tum": "colorectal adenocarcinoma epithelium",
    "tumor": "colorectal adenocarcinoma epithelium",
    "tumour": "colorectal adenocarcinoma epithelium",
    "adenocarcinoma": "colorectal adenocarcinoma epithelium",
    "colorectal adenocarcinoma": "colorectal adenocarcinoma epithelium",
}


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;!?\"'`*_")
    return text


def canonicalize(text: str) -> str:
    """Map a raw answer string onto a canonical class name where possible."""
    norm = normalize(text)
    if norm in CLASS_NAMES:
        return norm
    if norm in ALIASES:
        return ALIASES[norm]
    return norm


class NctCrcEnv(BaseTextEnv):
    """
    Single-turn environment for 9-class colorectal tissue classification.

    Interaction protocol:
        1. Model receives the patch image + the 9-class question (via init).
        2. Model reasons and emits <answer>class_name</answer>.
        3. Env scores the answer. Episode always ends after this one step.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = canonicalize(str(extras["reward_spec"]["ground_truth"]))

        # Single-turn by construction. `extras["max_turns"]` is set by the
        # generator from generator.max_turns and is deliberately ignored here.
        self.max_turns = 1
        self.correct = False
        self.well_formed = False

    def _extract_answer(self, text: str) -> str | None:
        """Return the contents of the last <answer>...</answer> block, if any."""
        matches = ANSWER_RE.findall(text)
        if not matches:
            return None
        return matches[-1].strip()

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        raw_answer = self._extract_answer(action)
        self.well_formed = raw_answer is not None and len(raw_answer) > 0
        parsed = canonicalize(raw_answer) if self.well_formed else None
        self.correct = parsed == self.ground_truth

        reward = (CORRECT_REWARD if self.correct else 0.0) + (FORMAT_BONUS if self.well_formed else 0.0)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={
                "ground_truth": self.ground_truth,
                "answer": parsed,
                "well_formed": self.well_formed,
                "correct": self.correct,
            },
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "acc": 1.0 if self.correct else 0.0,
            "format_ok": 1.0 if self.well_formed else 0.0,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        n = len(metrics)
        return {
            "avg_acc": sum(float(m.get("acc", 0)) for m in metrics) / n,
            "avg_format_ok": sum(float(m.get("format_ok", 0)) for m in metrics) / n,
        }
