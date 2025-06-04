.PHONY: setup build run clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt

build:
	. .venv/bin/activate && python utils/build_vectorstore.py

run:
	. .venv/bin/activate && python agent_app/main_graph.py

clean:
	rm -rf .venv vectorstore
