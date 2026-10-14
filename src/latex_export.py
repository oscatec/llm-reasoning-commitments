"""Render parsed reasoning traces as a LaTeX document with empty annotation slots."""

from __future__ import annotations

import re
from pathlib import Path

_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "\\": r"\textbackslash{}"}


def latex_escape(text: str) -> str:
    return re.sub(r"[&%$#_{}~^\\]", lambda m: _ESCAPES[m.group()], text)


def _entry(trace, idx: int) -> str:
    lines = [
        rf"\subsection*{{Question {idx}}}",
        rf"\noindent\textbf{{Question:}} {latex_escape(trace.question)}\\[0.5em]",
        r"\noindent\textbf{Options:}",
        r"\begin{itemize}[leftmargin=2em]",
        *[rf"  \item[{k}.] {latex_escape(v)}" for k, v in trace.options.items()],
        r"\end{itemize}",
        r"\noindent\textbf{Reasoning trace:}",
        r"\begin{enumerate}",
        *[rf"  \item {latex_escape(s['text'])}" for s in trace.steps if s["text"]],
        r"\end{enumerate}",
        rf"\noindent\textbf{{Model answer:}} {trace.solution or 'none'} \quad "
        rf"\textbf{{Ground truth:}} {trace.ground_truth}\\[0.5em]",
    ]
    if trace.format_issues:
        lines.append(rf"\noindent\textit{{Format issues: {latex_escape('; '.join(trace.format_issues))}}}\\[0.5em]")
    lines += [
        r"\noindent\textbf{Commitments}\\",
        r"\noindent Logical principle: \\[2em]",
        r"\noindent Explicit: \\[4em]",
        r"\noindent Implicit: \\[4em]",
        r"\newpage",
    ]
    return "\n".join(lines)


def write_latex(traces, path: Path, title: str, subtitle: str = "") -> None:
    doc = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{enumitem}",
        r"\usepackage{parskip}",
        rf"\title{{{latex_escape(title)}\\[0.5em]\large {latex_escape(subtitle)}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        *[_entry(t, i) for i, t in enumerate(traces, 1)],
        r"\end{document}",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(doc), encoding="utf-8")
