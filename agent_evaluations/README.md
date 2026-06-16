# agent_evaluations — AI conversion variance tools

PyEPQ is produced by a prompt-driven AI workflow ([docs/PROMPTS.md](../docs/PROMPTS.md)).
This tool answers one question about that workflow: **when the AI is given the
same prompt and context, how much do its code outputs vary?**

Low variance across regenerations is evidence the conversion process is stable
and reproducible. High variance flags a prompt that needs tightening.

## How it works

Drop several AI-generated conversions of the **same** class (i.e. multiple
responses to one prompt) into a folder. `compare_variants.py` scores every pair
of files with two metrics and aggregates the results:

| Metric | What it measures |
|---|---|
| **CodeBLEU** | Code-aware similarity: n-gram, weighted n-gram, AST/syntax match, dataflow match. Scored *symmetrically* (mean of both directions), since neither AI output is a "reference". |
| **ROUGE-N** | Token n-gram overlap on a code-aware token stream (comments and layout stripped). The reported figure is F1, which is symmetric. |

### CodeBLEU engine

Because every candidate is Python, the syntax and dataflow components are
computed from the standard-library `ast` module — **no `tree-sitter`, no C
compiler, no install required** (the `ast` engine, used by default). For Python
this is actually more accurate than the language-agnostic reference
implementation.

If you have the optional [`codebleu`](https://pypi.org/project/codebleu/) package
working, run with `--engine codebleu` to use it instead. `--engine auto` (the
default) uses `codebleu` when it imports and otherwise falls back to the built-in
`ast` engine. The chosen engine is recorded in the report.

The pairwise scores are aggregated into **mean / stdev / min / max**. Those
aggregates are the variance result:

- **High mean + low stdev** → the AI is reproducible for that prompt.
- **Low mean / high spread** → high variance between responses.

## Install

Nothing to install — the default engine uses only the Python standard library.

The reference `codebleu` package is **optional** and only needed for
`--engine codebleu`. It pulls in `tree-sitter`, which has no prebuilt wheel on
Python 3.14 and so needs a C compiler (MSVC Build Tools) to build:

```
pip install codebleu tree-sitter-python
```

## Usage

1. Put at least two `.py` variants in `candidates/` (or any folder).
2. Run:

```
python compare_variants.py                 # compares candidates/
python compare_variants.py path/to/folder  # compares another folder
```

Useful options:

```
--engine auto|ast|codebleu # CodeBLEU engine (default: auto)
--rouge-n 1 2 3            # ROUGE n-gram orders (default: 1 2)
--weights .25 .25 .25 .25  # CodeBLEU component weights: ngram, w-ngram, syntax, dataflow
--out path/to/reports      # output dir (default: reports/)
-v                         # print each pair as it is compared
```

## Output

Written to `reports/` (override with `--out`):

- `variance_<folder>.json` — full per-pair breakdown (all CodeBLEU components and
  ROUGE precision/recall/F1) plus the aggregate block. Machine-readable for
  aggregating variance across many prompt runs.
- `variance_<folder>.md` — human-readable report: aggregate summary table,
  pairwise table, and CodeBLEU component breakdown.

A one-line summary is also printed to the terminal.

## Agents under test (so far)

| Folder | Agent | Model |
|---|---|---|
| `candidates/ChatSRS(gpt4.1)/` | ChatSRS | GPT-4.1 |
| `candidates/Gemini(3.1Pro)/` | Google AI Studio | Gemini 3.1 Pro |
| `candidates/Gemini(2.5Pro)/` | Google AI Studio | Gemini 2.5 Pro |
| `candidates/Claude(Sonnet4.6Low)/` | Antropic Claude | Claude Sonnet 4.6 Low Effort |


Each folder contains responses produced by that agent to the same prompt and
context defined in [docs/PROMPTS.md](../docs/PROMPTS.md). The canonical
reference port (`AdaptiveRungeKutta_ver1_1_2.py`) is placed directly in
`candidates/` and is auto-detected by the tool — it is not an agent response.

## Scope

This tool measures **textual/structural similarity**, not behavioral
correctness. Whether a conversion is actually *right* is the job of the parity
suite and [tools/check_compliance.py](../tools/check_compliance.py). Two outputs
can be behaviorally identical yet score low here (different but equivalent code),
and vice versa — interpret the numbers as reproducibility, not validity.
