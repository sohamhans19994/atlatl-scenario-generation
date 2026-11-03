# Running experiments

This guide contains the implementation details intentionally kept out of the
research-focused project page. A full run requires a separately obtained,
authorized AtlAtl installation.

## Requirements and installation

Python 3.11 is required. Clone your authorized AtlAtl checkout next to this
repository, keeping the default folder names:

```text
<parent>/
├── atlatl_ICT/
└── atlatl-scenario-generation/
```

Then, from this repository:

```bash
uv sync          # installs this repo and ../atlatl_ICT, both editable
uv run pytest    # offline integration tests: fake LLM, real engine
```

`pyproject.toml` points uv at `../atlatl_ICT` (`[tool.uv.sources]`), so code
here can import anything from the engine (`server.*`, `models.*`), and edits in
either checkout take effect without reinstalling. Prefix commands with
`uv run`, or activate `.venv`.

To use an AtlAtl checkout somewhere else, install it into the same environment
instead:

```bash
uv pip install -e /authorized/path/to/atlatl_ICT
```

The installed AtlAtl package must expose `server` and `server.server`. The
scenario generator locates that package automatically and invokes it with the
same Python interpreter.

The AtlAtl package must not also ship its own `scenario_generation` package,
or it will shadow this one. `atlatl_ICT/pyproject.toml` must list only
`include = ["server*", "models*"]`; `tests/test_wiring.py` checks this.

For scenarios using AtlAtl's naval combat model, set:

```bash
export ATLATL_COMBAT=naval
```

## Model providers

### OpenAI

OpenAI is the default provider:

```bash
export OPENAI_API_KEY="..."

python -m scenario_generation.mind_evolution.mind_evolution \
  --config examples/config.yaml \
  --objective examples/objective.yaml \
  --output-dir runs \
  --model gpt-5
```

### llama.cpp

The pipeline can use a text model served through llama.cpp's OpenAI-compatible
API. Start `llama-server` with enough context for replay evaluation and obtain
the exact model ID from `/v1/models`:

```bash
curl http://127.0.0.1:8080/v1/models

python -m scenario_generation.mind_evolution.mind_evolution \
  --config examples/config.yaml \
  --objective examples/objective.yaml \
  --output-dir runs \
  --llm-provider llama-cpp \
  --llm-base-url http://127.0.0.1:8080/v1 \
  --model "<model-id-from-v1-models>" \
  --context-window 32768
```

No key is required for an unauthenticated local server. If the server requires
a bearer token, provide it through `LLAMA_CPP_API_KEY`.

Scenario-generation reasoning is controlled with `--thinking off`, `high`, or
`max`. When reasoning is enabled, `--max-output-tokens` includes reasoning and
the final response, so reasoning models may require a larger budget.

Example selection and replay evaluation use a separate non-reasoning client for
strict JSON output. Its default output limit is 4,096 tokens and can be changed
with `--structured-max-output-tokens`.

## Experiment inputs

`examples/objective.yaml` defines:

- the learning objective;
- the expected strategy;
- map- and replay-based evaluation questions;
- the expected answer datatype for each question.

`examples/config.yaml` controls the evolutionary search:

- `N_generations`: maximum generations;
- `N_islands`: independent populations;
- `N_convs`: conversations per island;
- `N_seqs`: refinement turns per conversation;
- `N_reset_interval` and `N_reset`: island resets;
- `N_top`: strong candidates used to seed reset islands;
- `N_parents` and `Pr_no_parents`: parent selection;
- `N_emigrate`: candidates migrated between islands.

The six reference tasks used for in-context example retrieval live in
`scenario_generation/mind_evolution/examples/examples.json`.

## Output files

Each experiment creates a timestamped directory under the selected output
directory:

```text
runs/<run-id>/
├── config.json
├── llm_config.json
├── objective.json
├── events.log
├── llm_results.jsonl
├── evaluation_results.jsonl
├── evolution_graph.json
├── evolution_results.json
└── replays/
```

Incremental candidate results may also be written while a run is active. Raw
logs and replays can be large and may contain complete prompts, so they are
ignored by Git.

## Map representation

Generated maps use a staggered text grid. Terrain is encoded with one
character:

| Code | Terrain |
| --- | --- |
| `c` | clear |
| `u` | urban |
| `r` | rough land |
| `m` | marsh |
| `w` | deep water |
| `s` | coast or shore |

An optional unit token contains strength, type, and faction. For example,
`100ab` represents strength-100 Blue artillery. Generated structure is
validated before conversion into an AtlAtl scenario.

## Standalone replay metrics

Basic replay metrics can be calculated separately:

```bash
python -m scenario_generation.replay_analysis.basic_metrics \
  --blue_replay /path/to/bluereplay.js \
  --red_replay /path/to/redreplay.js
```

The command reports phase count, Blue damage dealt, Blue damage taken, and the
dealt-to-taken ratio for each repetition.

## Practical limitations

- Full execution is impossible without an authorized AtlAtl installation.
- Generation and replay evaluation can require long model contexts.
- Model calls can be slow or incur API costs.
- LLM checklist answers are imperfect and should be manually reviewed.
- The compact converter supports the terrain and unit types documented by the
  prompt templates.
- Generated scenarios should be reviewed before training or analysis.
