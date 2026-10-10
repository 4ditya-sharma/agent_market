.PHONY: install data train train-quick train-transformer api test demo

install:
	pip install -r requirements.txt

data:            ## build 3-class dataset (pulls HF datasets if reachable, else synthetic only)
	python -m classifier.build_dataset

train:           ## full tuned sklearn firewall (~10-20 min)
	python -m classifier.train_model

train-quick:     ## sanity run (~2 min)
	QUICK_RUN=1 python -m classifier.train_model

train-transformer:  ## optional DeBERTa fine-tune (needs torch/transformers, GPU or Colab recommended)
	python -m classifier.train_transformer

api:
	uvicorn api.main:app --port 8001 --reload

test:
	LLM_MODE=mock python -m pytest -q tests

demo:
	python -m scripts.demo
