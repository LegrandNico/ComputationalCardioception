# save output to a log file
exec > output.log 2>&1

uv run python code/run.py --session 1 --model all --overwrite
uv run python code/run.py --session 2 --model all --overwrite