<!-- Keep this description current as the PR is worked on. For example, when a human reviews
the PR, say so in the AI tools used section. -->

## Summary

<!-- A few sentences: what this PR does and what a reviewer should take away. -->


## Why are we making this change?

<!-- The problem or goal that motivates the change. Link issues or the stack this PR belongs to. -->

## What changes were made?

<!-- The changes, grouped by area. Name the files or modules a reviewer should read first. -->

## Impacts to users

<!-- Changes to inputs, outputs, defaults, numerical results, dependencies or the public API. Write "None" if there are none, and say how you know. -->

## Checks ran

- [ ] `pytest` is green
- [ ] Regression tests pass (`poetry run python tests/utils/regression_results.py` regenerates the references; `poetry run pytest --no-cov tests/test_regression_against_cases.py` checks them): results are unchanged, or the source of every change is expected and the changes are summarised under "Impacts to users"
- [ ] Coverage reported (`pytest --cov`) and at or above the `--cov-fail-under` value in the CI workflow
- [ ] Tests written for new features and for the edge cases they handle
- [ ] `pre-commit run --all-files` is clean (`ruff check cfspopcon`, `ruff format`, `ty check`; the same checks run in CI)
- [ ] Complexity within the ruff `C901` limit (`max-complexity` in `pyproject.toml`); any `# noqa: C901` is justified
- [ ] `ty check` is clean
- [ ] No Dependabot alerts left open by this PR, or the open ones are named below
- [ ] Documentation written and/or updated (README, docs, docstrings, CHANGELOG)

<!-- Paste the pytest summary line and the total coverage figure. For any box left unticked, say why. -->

## Evidence that the changes achieve the goal

<!-- Run-times, stored-reference-value or bit-identity results, validation comparisons.
Prefer figures: attach plots and state the command that produced them. -->

## Follow-on work

<!-- Things that should be done but are not in this PR. -->

-

## AI tools used

<!-- A few sentences: which tools, and how the work was split between human and AI at each
stage (idea, implementation, review). Say which parts of the output a reviewer should read
most closely. Update this as the PR changes: if a human reviews it, or the work is revised,
say who did what. For new contributors this also shows what use of AI is acceptable here. -->
