# Coding Skill — Red-Green TDD

Follow strict Red-Green-Refactor TDD for all code changes in this project.

## Workflow

### 1. RED — Write a Failing Test First

- Before writing any production code, write a test that captures the desired behavior.
- Place tests in `tests/` mirroring the source layout (e.g., `src/ck_trading/strategies/foo.py` → `tests/test_strategies/test_foo.py`).
- Use `pytest` with the existing fixtures from `tests/conftest.py` (`sample_prices`, `sample_fundamentals`, `tmp_data_dir`).
- Run the test with `uv run pytest <test_file>::<test_name> -x` and confirm it **fails** (red).
- If the test passes immediately, the test is not capturing new behavior — revise it.

### 2. GREEN — Write Minimal Code to Pass

- Write the **simplest** production code that makes the failing test pass.
- Do not add extra logic, optimizations, or features beyond what the test requires.
- Run the test again with `uv run pytest <test_file>::<test_name> -x` and confirm it **passes** (green).

### 3. REFACTOR — Clean Up While Green

- With tests passing, improve code quality: remove duplication, improve naming, simplify logic.
- Run the **full related test file** with `uv run pytest <test_file> -x` to ensure nothing breaks.
- Do not change behavior during refactoring — all tests must stay green.

### 4. Repeat

- Continue the cycle for each new piece of behavior. One behavior per cycle.

## Rules

- **Never write production code without a failing test first.**
- **Never skip the red step.** Always verify the test fails before making it pass.
- **One assertion focus per test.** Tests should be specific — test one behavior, not everything at once. Multiple `assert` statements are fine if they validate the same behavior.
- **Keep tests fast.** No network calls, no disk I/O beyond `tmp_path`. Mock external dependencies.
- **Test naming:** `test_<unit>_<scenario>` (e.g., `test_graham_filters_high_pe`).
- **Show your work.** After each RED and GREEN step, report the test command and its result (pass/fail) so the user can follow the TDD cycle.

## Test Commands

```bash
# Run a single test (RED/GREEN step)
uv run pytest tests/<path>::<test_name> -x

# Run a test file (REFACTOR step)
uv run pytest tests/<path> -x

# Run full suite (before commit)
uv run pytest --tb=short

# Run with coverage
uv run pytest --cov=src/ck_trading --tb=short
```

## Project Conventions

- **Language:** Python 3.12+
- **Test framework:** pytest (with pytest-asyncio for async code)
- **Data library:** Polars (not pandas)
- **Linting:** `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/`
- **Type checking:** `uv run pyright src/`
- **Source root:** `src/ck_trading/`
- **Test root:** `tests/`

## Before Committing

1. Run `uv run ruff check src/ tests/` — fix any lint errors.
2. Run `uv run ruff format src/ tests/` — format code.
3. Run `uv run pytest --tb=short` — full test suite must pass.
