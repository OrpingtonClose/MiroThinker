# Eval suite for the Strands Venice agent (MiroThinker)

.PHONY: test test-offline test-integ test-judge test-all lint

# Run offline evals only (no API keys needed)
test: test-offline

test-offline:
	cd .. && python -m pytest evals/ -v -m "not integ" --override-ini="addopts="

# Run integration evals (requires VENICE_API_KEY)
test-integ:
	cd .. && python -m pytest evals/test_integ_agent.py -v -m integ --override-ini="addopts="

# Run LLM-as-judge evals (requires VENICE_API_KEY)
test-judge:
	cd .. && python -m pytest evals/test_integ_llm_judge.py -v -m integ --override-ini="addopts="

# Run everything
test-all:
	cd .. && python -m pytest evals/ -v --override-ini="addopts="

# Lint eval files
lint:
	cd .. && python -m ruff check evals/
	cd .. && python -m ruff format --check evals/
