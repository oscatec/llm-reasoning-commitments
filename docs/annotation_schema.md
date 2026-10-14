# Annotation schema

The aim is to separate *what a model says* from *what its reasoning depends on*. Each trace is annotated with three layers of commitment, plus quality labels.

## Commitment layers

**Logical principle.** The single rule that licenses the final answer, written as a general statement rather than a restatement of the answer. It answers the question "what would have to be true for this reasoning to work?"

**Explicit commitments.** Factual claims the model states in the trace, kept close to its own wording, one claim per item. These are checkable against domain knowledge independently of whether the final answer is right.

**Implicit commitments.** Steps the reasoning relies on but never states: why an option is eliminated, an unstated assumption that links two claims, or a background fact that the conclusion needs.

## Worked example

> **Question.** In fungi, cell walls are reinforced by which of the following?
> A. Chitin  B. Silica  C. Lignin  D. Collagen fibers
>
> **Trace (Qwen2.5 7B, abridged).** Step 1: Chitin is a major component of fungal cell walls and provides structural support. Step 2: Silica is often found in plant cell walls but not typically in fungal cell walls. Step 3: Lignin is primarily found in the secondary cell walls of vascular plants. Step 4: Collagen fibers are found in animal tissues and are not present in fungi. Step 5: Chitin is the correct component. Solution: Choice_A.

| Layer | Annotation |
|---|---|
| Logical principle | The reinforcing component is the one present in fungal walls when the alternatives belong to other lineages. |
| Explicit | Chitin is a major component of fungal cell walls; chitin provides structural support; silica occurs in plant cell walls, not typically in fungal ones; lignin occurs in the secondary walls of vascular plants; collagen occurs in animal tissue and not in fungi. |
| Implicit | B and C are eliminated because silica and lignin are plant features; D is eliminated because collagen is found in animal tissue; chitin microfibrils provide the tensile strength that fungal cells need to keep their shape. |

The full annotated sets are in [`examples/`](examples/).

## Quality labels

| Field | Values | Meaning |
|---|---|---|
| `model_answer_is_correct` | yes / no | Final answer matches the ground truth. A trace with no explicit final answer counts as incorrect. |
| `trace_is_valid` | yes / no | The trace follows the requested format: numbered steps, every option addressed, one final `Solution:` line. |
| `annotation_confidence` | high / medium / low | How sure the annotator is that the explicit list is complete and faithful. |
| `annotator_notes` | free text | Reason for a validity failure or anything unusual. |

## Fields in `data/annotations.csv`

| Column | Description |
|---|---|
| `model_name`, `mode` | Model that produced the trace and the prompting condition (`independent` or `shared`) |
| `subject`, `question_uuid`, `question`, `option_A` to `option_D` | The Mr-Ben item |
| `analysis_json`, `analysis_text` | The trace, as a list of steps and as plain text |
| `solution`, `solution_letter`, `ground_truth` | Model answer and correct answer |
| `annotator_explicit_reasoning` | Explicit commitments, one per line |
| `model_answer_is_correct`, `trace_is_valid`, `annotation_confidence`, `annotator_notes` | Quality labels above |
