setup-env:
	sudo apt-get install -y graphviz
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

run-notebooks:
	@echo "--- 📓 Running notebooks ---"
	uv run --with jupyter jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.kernel_name=python3 \
		code/notebooks/*.ipynb
