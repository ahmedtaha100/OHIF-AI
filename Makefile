PYTHON ?= python
VENV ?= .venv
VENV_PY := $(shell $(PYTHON) -c "import os; print('\"$(VENV)/Scripts/python.exe\"' if os.name == 'nt' else '$(VENV)/bin/python')")

NNINTERACTIVE_SERVER_URL ?= http://127.0.0.1:1527
BLACKWELL_MODEL ?= nnInteractive_v1.0
BLACKWELL_DEVICE ?= cuda:0
BLACKWELL_MAX_SESSIONS ?= 1
BLACKWELL_ENV_COUNT ?= 1
BLACKWELL_INTERACTIONS_STORAGE ?= blosc2
BLACKWELL_DATASET_MANIFEST ?= manifests/blackwell_datasets.json
BLACKWELL_OUTPUT_DIR ?= artifacts/rl_nninteractive/blackwell_handoff
BLACKWELL_DQN_EPISODES ?= 256
BLACKWELL_MAX_REMOTE_ENV_STEPS ?= 10000
THROUGHPUT_IMAGE_ARGS ?= --use-nibabel-test-image
THROUGHPUT_PARALLEL_SESSIONS ?= 1

.PHONY: setup setup-real test smoke real-smoke verify-baseline verify-baseline-public phase1-small phase1-real phase2-smoke throughput-remote blackwell-server blackwell-handoff clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.lock
	$(VENV_PY) -m pip install -e .

setup-real:
	$(VENV_PY) -m pip install -r requirements.real.txt

test:
	$(VENV_PY) -m unittest discover -s tests -p "test_*.py"

smoke:
	$(VENV_PY) -m rl_nninteractive --config configs/rl_nninteractive_skeleton.json

real-smoke:
	$(VENV_PY) -m rl_nninteractive.real_smoke --require-real --use-nibabel-test-image --output-dir artifacts/rl_nninteractive/real_smoke

verify-baseline:
	$(VENV_PY) -m rl_nninteractive.verify_baseline --output-dir artifacts/rl_nninteractive/baseline_verification/synthetic

verify-baseline-public:
	$(VENV_PY) -m rl_nninteractive.verify_baseline --include-public-nibabel --output-dir artifacts/rl_nninteractive/baseline_verification/public_nibabel

phase1-small:
	$(VENV_PY) -m rl_nninteractive.phase1_small --output-dir artifacts/rl_nninteractive/phase1_small

phase1-real:
	$(VENV_PY) -m rl_nninteractive.phase1_real --require-remote --dataset-manifest $(BLACKWELL_DATASET_MANIFEST) --server-url $(NNINTERACTIVE_SERVER_URL) --output-dir artifacts/rl_nninteractive/phase1_real --dqn-episodes $(BLACKWELL_DQN_EPISODES) --max-remote-env-steps $(BLACKWELL_MAX_REMOTE_ENV_STEPS)

phase2-smoke:
	$(VENV_PY) -m rl_nninteractive.phase2_smoke --output-dir artifacts/rl_nninteractive/phase2_smoke

throughput-remote:
	$(VENV_PY) -m rl_nninteractive.throughput --require-remote --server-url $(NNINTERACTIVE_SERVER_URL) $(THROUGHPUT_IMAGE_ARGS) --parallel-sessions $(THROUGHPUT_PARALLEL_SESSIONS) --server-max-sessions $(BLACKWELL_MAX_SESSIONS) --output-dir artifacts/rl_nninteractive/throughput_remote

blackwell-server:
	"$(VENV)/Scripts/nninteractive-server.exe" --model $(BLACKWELL_MODEL) --host 127.0.0.1 --port 1527 --device $(BLACKWELL_DEVICE) --max-sessions $(BLACKWELL_MAX_SESSIONS) --no-torch-compile --interactions-storage $(BLACKWELL_INTERACTIONS_STORAGE)

blackwell-handoff:
	$(VENV_PY) -m rl_nninteractive.blackwell_handoff --server-url $(NNINTERACTIVE_SERVER_URL) --dataset-manifest $(BLACKWELL_DATASET_MANIFEST) --output-dir $(BLACKWELL_OUTPUT_DIR) --model $(BLACKWELL_MODEL) --device $(BLACKWELL_DEVICE) --max-sessions $(BLACKWELL_MAX_SESSIONS) --env-count $(BLACKWELL_ENV_COUNT) --interactions-storage $(BLACKWELL_INTERACTIONS_STORAGE) --phase1-dqn-episodes $(BLACKWELL_DQN_EPISODES)

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', 'build', 'dist', 'ohif_ai_rl_nninteractive.egg-info')]"
