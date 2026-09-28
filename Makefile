PY ?= python3
.PHONY: verify kernels smoke replay figures bench campaign timing selftest
verify:
	$(PY) reproduce.py verify
kernels:
	$(PY) reproduce.py kernels
smoke:
	$(PY) reproduce.py smoke
replay:
	$(PY) reproduce.py replay
figures:
	$(PY) reproduce.py figures
bench:
	$(PY) reproduce.py bench
campaign:
	$(PY) reproduce.py campaign
timing:
	$(PY) reproduce.py timing
selftest:
	$(PY) reproduce.py selftest
