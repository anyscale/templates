"""
Rule rewards for NCT-CRC-HE tissue classification, as plain TRL reward functions.

Scoring is identical to the SkyRL arm (../biotech_vlm_grpo/env.py, NctCrcEnv.step):

    label_reward   1.0 if the last <answer>...</answer> block canonicalises to the
                   ground-truth class, else 0.0
    format_reward  0.2 if there is a non-empty <answer> block, else 0.0
                   (awarded even when the label is wrong)

    total          1.2 correct + well-formed, 0.2 wrong + well-formed, 0.0 no tags

TRL sums the reward functions (reward_weights default to 1.0 each), so the group
sees the same scalar the SkyRL env emits. They are two separate functions rather
than one so that grpo_step_timing.py can time each by name.

The answer-normalisation tables are copied verbatim from env.py rather than
imported: env.py depends on skyrl_gym, which is not (and should not be) in this
environment. Keep the two in sync if either changes.
"""

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

CORRECT_REWARD = 1.0
FORMAT_BONUS = 0.2

# Canonical class names, in the dataset's ClassLabel order
# (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM).
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
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\"'`*_")


def canonicalize(text: str) -> str:
    norm = normalize(text)
    if norm in CLASS_NAMES:
        return norm
    return ALIASES.get(norm, norm)


def extract_answer(text: str) -> str | None:
    """Contents of the last <answer>...</answer> block, or None."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _completion_text(completion) -> str:
    # Conversational datasets give [{"role": "assistant", "content": "..."}];
    # standard-format datasets give a plain string.
    if isinstance(completion, list):
        return "".join(str(m.get("content", "")) for m in completion)
    return str(completion)


def label_reward(completions, ground_truth, **kwargs) -> list[float]:
    """1.0 if the extracted answer matches the ground-truth class, else 0.0."""
    out = []
    for completion, gt in zip(completions, ground_truth):
        ans = extract_answer(_completion_text(completion))
        ok = ans is not None and ans != "" and canonicalize(ans) == canonicalize(str(gt))
        out.append(CORRECT_REWARD if ok else 0.0)
    return out


def format_reward(completions, **kwargs) -> list[float]:
    """0.2 if the completion has a non-empty <answer>...</answer> block, else 0.0."""
    out = []
    for completion in completions:
        ans = extract_answer(_completion_text(completion))
        out.append(FORMAT_BONUS if ans else 0.0)
    return out


if __name__ == "__main__":
    # The same 8 cases the SkyRL env was checked against.
    cases = [
        ("Looks like immune cells. <answer>lymphocytes</answer>", "lymphocytes", 1.2),
        ("<answer>debris</answer>", "lymphocytes", 0.2),
        ("I think lymphocytes but I am not sure.", "lymphocytes", 0.0),
        ("<answer>debris</answer> wait no <answer>lymphocytes</answer>", "lymphocytes", 1.2),
        ("<answer></answer>", "lymphocytes", 0.0),
        ("<answer>LYM</answer>", "lymphocytes", 1.2),
        ("<answer>Tumor.</answer>", "colorectal adenocarcinoma epithelium", 1.2),
        ("<ANSWER> Normal   Colon Mucosa </ANSWER>", "normal colon mucosa", 1.2),
    ]
    comps = [[{"role": "assistant", "content": c}] for c, _, _ in cases]
    gts = [g for _, g, _ in cases]
    total = [a + b for a, b in zip(label_reward(comps, gts), format_reward(comps))]
    for (c, g, want), got in zip(cases, total):
        flag = "ok " if abs(got - want) < 1e-9 else "BAD"
        print(f"{flag} want={want:.1f} got={got:.1f}  {c[:50]!r}")
    assert all(abs(t - w) < 1e-9 for t, (_, _, w) in zip(total, cases)), "reward regression"
    print("all reward cases pass")
