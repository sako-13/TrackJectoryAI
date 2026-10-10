.PHONY: install test pipeline pipeline-quick demo clean

install:
	pip3 install -r requirements.txt

test:
	python3 -m pytest

pipeline:
	python3 scripts/run_pipeline.py

pipeline-quick:
	python3 scripts/run_pipeline.py --quick

demo:
	python3 scripts/serve_demo.py

clean:
	rm -rf artifacts/* data/raw/* data/processed/* reports/figures/*
