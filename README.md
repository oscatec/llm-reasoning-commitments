# What does an LLM commit to when it reasons?

A small study of chain-of-thought traces from open LLMs on biology and medicine multiple-choice questions. Instead of only scoring the final answer, each trace is broken down into the commitments the model makes along the way: the logical principle behind its answer, the facts it states explicitly, and the steps it relies on without saying them.

The motivation is practical. A model can land on the right answer through reasoning that depends on a false premise, and a correct final answer hides that. Making the commitments explicit is the first step towards checking them.

![Accuracy and validity by model](figures/accuracy_validity.png)

## What is in this repository

| Path | Contents |
|---|---|
| `src/generate_traces.py` | Trace generation over [Mr-Ben](https://huggingface.co/datasets/Randolphzeng/Mr-Ben) with a Hugging Face backend (used on a SLURM GPU cluster) or a local Ollama backend, plus an automatic format checker |
| `src/latex_export.py` | Exports traces to LaTeX with empty commitment slots, ready for manual annotation |
| `docs/annotation_schema.md` | The annotation scheme with a worked example |
| `docs/examples/` | Annotated traces as PDFs, including logical principle, explicit and implicit commitments |
| `data/annotations.csv` | 50 annotated traces from Qwen3.6 27B, Llama 3.1 8B and Qwen2.5 7B across college biology, high school biology and college medicine |
| `analysis/analyze.py` | Statistical analysis with confidence intervals and exact tests |
| `analysis/results.md` | Full results |

## Main findings

These come from 50 traces, so the intervals are wide and the findings are descriptive. Details and caveats are in [`analysis/results.md`](analysis/results.md).

**Model size matters more for accuracy than for format.** Qwen3.6 27B answered 22 of 23 questions correctly; the two 7 to 8B models were around 60%. Llama 3.1 8B, however, followed the requested format almost as reliably as the larger model.

**Qwen2.5 7B struggles with the output format.** 5 of its 13 traces were malformed, against 2 of 37 for the other two models combined (Fisher p = 0.009). The typical failure was copying the literal label "Step n" from the prompt template instead of numbering its last step.

**Format failures were not reasoning failures.** All 7 malformed traces still reached the correct answer. A pipeline that discards malformed outputs would have thrown away correct answers, which argues for parsing leniently and validating format and content as separate checks. The format checker in `generate_traces.py` flags these cases automatically instead of dropping them.

**Models state far more than they need.** Every trace carried between 5 and 12 explicit factual commitments for a single four-option question, each of which is a separate opportunity for a factual error that the final answer does not reveal.

## Running it

```bash
pip install -r requirements.txt

# Generate traces locally with Ollama
python src/generate_traces.py --backend ollama --model qwen2.5:7b \
    --subject college_biology --n 10 --out traces/qwen.jsonl --latex traces/qwen.tex

# Or on a GPU with transformers
python src/generate_traces.py --backend hf --model meta-llama/Llama-3.1-8B-Instruct \
    --subject college_medicine --n 20 --out traces/llama.jsonl

# Reproduce the analysis and figure
python analysis/analyze.py
```

Llama models on Hugging Face require accepting Meta's licence first.

## Context

This started as my contribution to a course-wide study at the University of Groningen, in which students annotated traces from different models to compare how they use explicit and implicit knowledge across fields. The collective study was never published. This repository contains only my own part: the generation pipeline, my annotation set and the analysis, which I extended afterwards.

## Limitations

The sample is small and was not designed as a balanced experiment: models, subjects and prompting modes are unevenly represented, and all annotations come from a single annotator, so there is no inter-annotator agreement. Three Llama traces ended without an explicit final answer and are counted as incorrect even where the conclusion in the last step matches the ground truth, which slightly understates its accuracy.

Questions and options come from Mr-Ben (Zeng et al., 2024).

Óscar Ferrer Domingo
