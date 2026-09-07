.PHONY: install docs check-docs test quality check

# Add the component toolchain to the shared Content Agents environment.
install:
	cd ../.. && uv pip install -e "apps/usd_cli[dev]" \
	  --overrides apps/usd_cli/requirements/usd-exchange-override.txt

# Regenerate docs/cli-reference.md from the live CLI. Run after any command help change.
docs:
	bash scripts/gen-docs.sh

# Public documentation gate: only the generated command reference is public.
check-docs: docs
	git diff --exit-code docs/cli-reference.md \
	  || (echo "docs/cli-reference.md is stale — run 'make docs' and commit." && exit 1)

test:
	pytest

quality:
	python -m compileall -q src apps/ovrtx_rendering_api/service
	ruff check --select E9,F63,F7,F82,F811 src tests apps/ovrtx_rendering_api/service
	mypy --ignore-missing-imports --allow-untyped-defs src/usd_core/history.py src/usd_core/models.py

check: quality test check-docs
