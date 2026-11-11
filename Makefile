.PHONY: test demo cli serve
test:
	PYTHONPATH=src python3 -m pytest tests/ -q
demo:
	PYTHONPATH=src python3 scripts/demo.py
cli:
	PYTHONPATH=src python3 -m dlserve.cli --help
serve:
	PYTHONPATH=src python3 -m uvicorn dlserve.service:app --host 127.0.0.1 --port 8000
