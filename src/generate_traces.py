"""
Generate step-by-step reasoning traces for multiple-choice questions from Mr-Ben.

Each model is asked to analyse every option in its own step and to commit to a
single answer at the end. Traces are parsed into steps, checked against the
expected format, and written as JSON Lines (one trace per line). A LaTeX export
for manual annotation is available through `latex_export.py`.

Two backends are supported:
  hf      Hugging Face transformers pipeline (used on a SLURM GPU cluster)
  ollama  a local Ollama server, handy for quick runs on a laptop

Examples
--------
python src/generate_traces.py --backend hf --model meta-llama/Llama-3.1-8B-Instruct \
    --subject college_biology --n 20 --out traces/llama.jsonl

python src/generate_traces.py --backend ollama --model qwen2.5:7b \
    --subject college_medicine --n 10 --out traces/qwen.jsonl --latex traces/qwen.tex
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROMPT_TEMPLATE = """Please generate a step-by-step analysis for the following Question in the subject {subject}.
Question: {question}
Choice_A: {A}
Choice_B: {B}
Choice_C: {C}
Choice_D: {D}

Here is the desired format, please analyse each candidate choice sequentially and then jointly decide which option is the solution in the final step.
Please ensure every newline character follows a step indicator:
Step 1: [The first reasoning of the step by step analysis on the candidate choices here]
Step 2: [The second reasoning of the step by step analysis on the candidate choices here]
...
Step n: [Conclude your analysis and decide which choice to make here]
Solution: Choice_A/B/C/D
Please follow this format without any additional introductory or concluding statements."""

STEP_RE = re.compile(r"^Step\s+(\d+|n)\s*:\s*(.*)$", re.IGNORECASE)
SOLUTION_RE = re.compile(r"Solution:\s*Choice_([A-D])", re.IGNORECASE)
OPTION_RE = re.compile(r"([A-D]):\s*(.*?)(?=\s+[A-D]:|$)", re.DOTALL)


@dataclass
class Trace:
    question_uuid: str
    subject: str
    question: str
    options: dict
    ground_truth: str
    model: str
    raw_output: str
    steps: list = field(default_factory=list)
    solution: str | None = None
    format_issues: list = field(default_factory=list)

    @property
    def is_correct(self) -> bool:
        return self.solution == self.ground_truth


def parse_options(options_str: str) -> dict:
    return {k: v.strip() for k, v in OPTION_RE.findall(options_str)}


def parse_trace(text: str) -> tuple[list, str | None, list]:
    """Split raw model output into steps and a solution, and record format problems.

    The issues flagged here are the ones that came up during manual annotation:
    literal "Step n" labels instead of numbers, missing final solution (usually a
    truncated output), and text outside the requested format.
    """
    steps, issues, current = [], [], None
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = STEP_RE.match(line)
        if m:
            if m.group(1).lower() == "n":
                issues.append("literal 'Step n' label")
            current = {"label": m.group(1), "text": m.group(2)}
            steps.append(current)
        elif line.startswith("Solution:"):
            continue
        elif current is not None:
            current["text"] += " " + line
        else:
            issues.append("text before first step")

    sol = SOLUTION_RE.search(text)
    if not sol:
        issues.append("no final solution (possibly truncated)")
    return steps, sol.group(1).upper() if sol else None, sorted(set(issues))


class HFBackend:
    def __init__(self, model: str, max_new_tokens: int):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model, device_map="auto")
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, return_full_text=False)
        return out[0]["generated_text"]


class OllamaBackend:
    def __init__(self, model: str, max_new_tokens: int, host: str = "http://localhost:11434"):
        import requests
        self.requests, self.model, self.host = requests, model, host
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        r = self.requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"num_predict": self.max_new_tokens}},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["response"]


def load_questions(subject: str, n: int, stride: int):
    from datasets import load_dataset
    ds = load_dataset("Randolphzeng/Mr-Ben", split="train")
    rows = [r for r in ds if r["Subject"] == subject]
    return rows[::stride][:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--subject", default="college_biology",
                    help="Mr-Ben subject, e.g. college_biology, high_school_biology, college_medicine")
    ap.add_argument("--n", type=int, default=10, help="number of questions")
    ap.add_argument("--stride", type=int, default=3, help="take every k-th question for variety")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--out", type=Path, required=True, help="output .jsonl")
    ap.add_argument("--latex", type=Path, help="optional LaTeX export for annotation")
    args = ap.parse_args()

    backend = (HFBackend if args.backend == "hf" else OllamaBackend)(args.model, args.max_new_tokens)
    subject_label = args.subject.replace("_", " ").title()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    traces = []
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(load_questions(args.subject, args.n, args.stride), 1):
            options = parse_options(row["Options"])
            prompt = PROMPT_TEMPLATE.format(subject=subject_label, question=row["Question"],
                                            **{k: options.get(k, "N/A") for k in "ABCD"})
            raw = backend(prompt)
            steps, solution, issues = parse_trace(raw)
            t = Trace(row["Question_UUID"], args.subject, row["Question"], options,
                      row["Ground_Truth_Answer"], args.model, raw, steps, solution, issues)
            traces.append(t)
            f.write(json.dumps(asdict(t) | {"is_correct": t.is_correct}, ensure_ascii=False) + "\n")
            print(f"[{i}/{args.n}] {t.question_uuid[:8]} answer={solution} "
                  f"truth={t.ground_truth} issues={issues or 'none'}")

    if args.latex:
        from latex_export import write_latex
        write_latex(traces, args.latex, title=f"Reasoning traces: {args.model}", subtitle=subject_label)

    correct = sum(t.is_correct for t in traces)
    clean = sum(not t.format_issues for t in traces)
    print(f"\nDone. {correct}/{len(traces)} correct, {clean}/{len(traces)} well-formed. Saved to {args.out}")


if __name__ == "__main__":
    main()
