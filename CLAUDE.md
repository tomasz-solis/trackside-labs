# trackside-labs agent guide

## Repo hygiene

End every session with a clean `git status`. Every file is committed, gitignored or deleted. Untracked files survive nothing.

Any path written at runtime gets its `.gitignore` entry in the same change. Generated artifacts belong in `ArtifactStore`, not git. Tracking one on purpose is a decision for the commit message.

Two traps already hit:

- `*.backup` does not match `<name>.json.rebuild_backup`. Check a new pattern with `git check-ignore -v <path>`.
- Tracked generated data goes stale. `data/car_characteristics_snapshot/` once tracked old files while the live ones were untracked.

Scoped but unbuilt work (briefs, tests for missing helpers, shelved research) goes on a `shelved/*` branch, for example `shelved/challenger-research` and `shelved/dnf-calibration`.

## Checks before claiming done

Ruff is pinned to one version (`0.15.1`) in three places. Bump all three together:

- `pyproject.toml` (`ruff==0.15.1`)
- `.pre-commit-config.yaml` (`rev: v0.15.1`)
- `.github/workflows/lint.yml` (`pip install ruff==0.15.1`)

```bash
uv sync --extra dev
uv run ruff check src tests scripts app.py predict_weekend.py
uv run ruff format --check src tests scripts app.py predict_weekend.py
make typecheck MYPY=mypy
```

CI runs tests in chunks over tracked files only. A bare `uv run pytest` also collects untracked files and can fail at collection. Use `make test-github-chunk-N`.

## Measuring model changes

- Results and verdicts live in `docs/MODEL_LEDGER.md`. Add new entries and mark old ones superseded; never change a past verdict.
- Every measured claim names the baseline it was measured against. The prediction cache key does not include the code version, so a cached prediction can predate the model it is credited to.
- Measure on the walk-forward replay with `--through-round`, against a rebuilt baseline and the seed floor. The protocol is at the top of the ledger.
- A 14-round replay takes 30 minutes to 3 hours (it varies with machine load) and about 1.2 GB. Run it in your own terminal: Claude Code's low-memory reaper has killed it as a background job.

## Writing docs

Keep it simple and short. Lead with the answer or result, say each thing once, keep the numbers and drop the story around them. No em or en dashes and no other AI tells. Tables for comparisons, lists for steps.

## House conventions

- uv first: `uv sync --extra dev`, `uv run <cmd>`.
