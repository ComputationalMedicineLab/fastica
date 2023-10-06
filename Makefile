.PHONY: inplace test test-all clean


all: clean inplace test


inplace:
	python -m setup build_ext --inplace

# Runs all the normal tests (some are very slow or use huge RAM)
test:
	python -m test_fastica

# Runs *all* the tests
test-all:
	FASTICA_TEST_ALL=1 python -m test_fastica

clean:
	rm -rf build/ __pycache__
	rm -f fastica.c fastica.html fastica.*.so

install:
	pip install . --no-build-isolation

uninstall:
	yes | pip uninstall fastica
