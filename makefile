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

comparison:
	uv run python code/comparison.py

# Number of notebooks executed concurrently (JOBS=1 restores serial execution)
JOBS ?= 4

run-notebooks:
	@echo "--- 📓 Running notebooks ($(JOBS) in parallel) ---"
	@ls code/notebooks/*.ipynb | xargs -P $(JOBS) -I {} \
		uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.kernel_name=python3 {}