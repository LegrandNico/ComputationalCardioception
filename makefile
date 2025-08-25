setup-env:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv
	bash source .venv/bin/activate 
	uv sync

run-jupyterlab:
	uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=computational_cardioception
	uv run --with jupyter jupyter lab --port 4444

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
