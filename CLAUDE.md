# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The canonical agent brief lives in [AGENTS.md](AGENTS.md) — repo orientation, code layout, single/multi-node launch, configuration, metrics/checkpoints/eval, and step-by-step extension instructions for new algorithms, models, and environments. The full contribution/PR flow is in [CONTRIBUTING.md](CONTRIBUTING.md). This file only adds the Claude-specific pointers that aren't in those files.

## Mental model

RLinf is a distributed RL stack for Embodied and Agentic AI. One launcher (e.g. [examples/embodiment/train_embodied_agent.py](examples/embodiment/train_embodied_agent.py)) builds a **Ray Cluster**, computes **component placement** (actor / rollout / env / reward / agent across nodes & GPUs), and starts **Worker groups**. A **Runner** drives the loop: `rollout → reward → advantage → actor update`. Training backends: FSDP or Megatron. Rollout: SGLang or vLLM. Loss/advantage/reward functions are selected by name from registries via Hydra config.

Three layers are worth keeping in mind before editing:
- **Config** ([rlinf/config.py](rlinf/config.py)): `build_config` / `validate_cfg` assemble the DictConfig. New model/env types must be added to `SupportedModel` / `SupportedEnvType` *and* the matching validation branch.
- **Scheduler** ([rlinf/scheduler/](rlinf/scheduler/)): `Cluster`, `Worker`, `WorkerGroup`, channels, placement strategies. Workers are Ray remote actors that `send`/`recv` across groups. Multi-node requires `RLINF_NODE_RANK` set **before** `ray start` on each node — Ray captures env at start time.
- **Registries** ([rlinf/algorithms/registry.py](rlinf/algorithms/registry.py), [rlinf/algorithms/rewards/__init__.py](rlinf/algorithms/rewards/__init__.py)): advantages, policy losses, and rewards are looked up by name. `algorithm.adv_type` and `algorithm.loss_type` in YAML drive dispatch.

## Common commands

**Install (single machine):**
```bash
bash requirements/install.sh embodied --model <model> --env <env>   # set REPO_PATH and asset paths first
```
Targets: `embodied`, `reason`, `docs`. See [requirements/install.sh](requirements/install.sh) for the full `--model`/`--env` matrix.

**Lint & format (must pass CI):**
```bash
pip install pre-commit
pre-commit install --hook-type commit-msg
pre-commit run --all-files                          # Ruff lint+format, commit-msg & sign-off checks
```

**Unit tests:**
```bash
export PYTHONPATH=$(pwd):$(pwd)/tests/unit_tests
pytest tests/unit_tests                             # full suite
pytest tests/unit_tests/test_worker.py              # single file
pytest tests/unit_tests/test_worker.py::TestName    # single test
pytest --doctest-modules rlinf/scheduler            # scheduler doctests (run in CI)
```
GPU-required tests run on the `reason` CI runner; many will skip without CUDA. CPU-only tests run on `ubuntu-latest`.

**E2E configs** live in [tests/e2e_tests/](tests/e2e_tests/) (embodied, agent, reasoning, sft, scheduler, dynamic_scheduler, auto_placement, training_backend). Each `.yaml` under `embodied/` is invoked by the matching CI workflow in [.github/workflows/](.github/workflows/).

**Launch training (single node):**
```bash
ray start --head                                                    # or let it auto-start
bash examples/embodiment/run_embodiment.sh <config_name>            # or use the Python entry directly:
python examples/embodiment/train_embodied_agent.py --config-name <config_name>
```
Examples per domain: [examples/embodiment/](examples/embodiment/), [examples/reasoning/](examples/reasoning/), [examples/sft/](examples/sft/), [examples/agent/](examples/agent/), [examples/recap/](examples/recap/), [examples/reward/](examples/reward/). Embodied configs often need env vars like `MUJOCO_GL=egl` or `ROBOT_PLATFORM` — check the launch script.

**Multi-node:** see AGENTS.md → "Single-node and multi-node" for the exact `RLINF_NODE_RANK` + `ray start --head` / `ray start --address=...` sequence, plus [ray_utils/start_ray.sh](ray_utils/start_ray.sh).

## Repository-specific conventions

- **No `print`.** In a `Worker` use `self.log_info` / `log_warning` / `log_error`; elsewhere `from rlinf.utils.logging import get_logger`.
- **YAML config rules** (enforced socially, not by linter): static values only, no computed fields, do not overwrite user-facing fields in code, avoid cross-field references.
- **Commits:** Conventional Commits, ~72-char subject, imperative, and **every commit must be `Signed-off-by:`** (use `git commit -s`). PR titles follow the same format. Pre-commit's `commit-check` enforces both — don't bypass with `--no-verify`.
- **Skills available** for repetitive scaffolding when adding new pieces: `add-install-docker-ci-e2e` (install script + Dockerfile stage + CI job + e2e config), `add-example-doc-model-env` (RST docs gallery EN+ZH), `docs-check` (cross-check EN/ZH docs against code), `review-pr` (CONTRIBUTING.md compliance), `add-publication-docs`. Prefer these over hand-rolling boilerplate.

## Extending: where to register

| Add a... | Touch these |
|---|---|
| Advantage fn | [rlinf/algorithms/advantages.py](rlinf/algorithms/advantages.py) + `@register_advantage("name")`; YAML `algorithm.adv_type` |
| Policy loss | [rlinf/algorithms/losses.py](rlinf/algorithms/losses.py) + `@register_policy_loss("name")`; YAML `algorithm.loss_type` |
| Reward | class under [rlinf/algorithms/rewards/](rlinf/algorithms/rewards/) + `register_reward("name", Cls)` in `__init__.py` |
| Embodied model | `SupportedModel` enum in [rlinf/config.py](rlinf/config.py) + package under [rlinf/models/embodiment/](rlinf/models/embodiment/) (inherit `BasePolicy`) + branches in actor/rollout workers + install/Docker/CI |
| Environment | `SupportedEnvType` + `get_env_cls()` branch in [rlinf/envs/__init__.py](rlinf/envs/__init__.py) (lazy import) + package under [rlinf/envs/](rlinf/envs/) + maybe `prepare_actions` branch in [rlinf/envs/action_utils.py](rlinf/envs/action_utils.py) + install/Docker/CI |
| New task type | new runner under [rlinf/runners/](rlinf/runners/) + entry script under `examples/` that builds Cluster, placement, worker groups, calls the runner |

AGENTS.md has the full step-by-step for each row; this table is the quick lookup.
