# save output to a log file
exec > output.log 2>&1

uv run python code/run.py --session 1 --model cardiac_hgf
uv run python code/run.py --session 2 --model cardiac_hgf