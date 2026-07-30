setup-env:
	# Graphviz *binaries* (dot) are required by pm.model_to_graphviz
	if [ "$$(uname)" = "Darwin" ]; then \
		brew install graphviz; \
	else \
		sudo apt-get update && sudo apt-get install -y graphviz; \
	fi
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv
	bash source .venv/bin/activate 
	uv sync

jupyterlab:
	uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=computational_cardioception
	uv run --with jupyter jupyter lab --port 5555

pre-commit:
	uv run pre-commit install
	uv run pre-commit autoupdate
	uv run pre-commit run --all-files

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 						        # running ruff formatting
	uv run ruff check **/*.py --fix						# running ruff linting

run-models:
	@echo "--- 🧪 Running models ---"
	bash scripts.sh

merge:
	uv run python code/merge.py

# Number of notebooks executed concurrently (JOBS=1 restores serial execution)
JOBS ?= 4

NOTEBOOKS := $(wildcard code/notebooks/*.ipynb)
NB_LOGDIR := logs/notebooks

run-notebooks:
	@echo "--- 📓 Running notebooks ($(JOBS) in parallel) ---"
	@if [ ! -d results/idata ] || [ -z "$$(ls -A results/idata 2>/dev/null)" ]; then \
		echo "⚠️  results/idata/ is missing or empty."; \
		echo "   Notebooks that read fitted traces (model comparison, correlations)"; \
		echo "   will fail. Run 'make run-models' first, or restore the .nc files."; \
		echo ""; \
	fi
	@rm -rf $(NB_LOGDIR) && mkdir -p $(NB_LOGDIR)
	@printf '%s\n' $(NOTEBOOKS) | xargs -P $(JOBS) -I{} sh -c '\
		nb="$$0"; log="$(NB_LOGDIR)/$$(basename "$$nb" .ipynb).log"; \
		if uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace \
				--ExecutePreprocessor.kernel_name=python3 "$$nb" > "$$log" 2>&1; then \
			echo "  ✅ $$(basename "$$nb")"; \
		else \
			echo "  ❌ $$(basename "$$nb")"; \
			echo "$$nb" >> $(NB_LOGDIR)/FAILED; \
		fi' {}
	@if [ -s $(NB_LOGDIR)/FAILED ]; then \
		echo ""; \
		echo "--- ❌ $$(wc -l < $(NB_LOGDIR)/FAILED | tr -d ' ') notebook(s) FAILED ---"; \
		while read -r nb; do \
			echo ""; \
			echo "▸ $$nb"; \
			tail -n 25 "$(NB_LOGDIR)/$$(basename "$$nb" .ipynb).log" | sed 's/^/    /'; \
		done < $(NB_LOGDIR)/FAILED; \
		echo ""; \
		echo "Full logs in $(NB_LOGDIR)/."; \
		echo "NOTE: nbconvert --inplace does not write on failure, so these notebooks"; \
		echo "      still contain their PREVIOUS outputs. Any figure or results file"; \
		echo "      they produce is now STALE, not regenerated."; \
		exit 1; \
	fi
	@echo ""
	@echo "--- ✅ all $(words $(NOTEBOOKS)) notebooks executed ---"