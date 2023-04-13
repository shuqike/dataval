PYTHON = python3

.DEFAULT_GOAL = help

help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project type make setup"
	@echo "To test the project type make test"
	@echo "To run the project type make run"
	@echo "------------------------------------"

setup: requirements.txt
    pip install -r requirements.txt

run:
	${PYTHON} run.py

test:
    ${PYTHON} test.py

clean:
    rm -rf __pycache__
