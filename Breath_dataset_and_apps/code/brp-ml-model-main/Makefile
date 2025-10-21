black:
	black .

linter:
	flake8 --exclude=venv --ignore=E501 .

format:
	black .
	isort .
	flake8 --exclude=venv --ignore=E501 .


.PHONY: black linter format