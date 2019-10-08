SHELL := /bin/bash
PROJECT=$(notdir $(shell pwd))
TENSORBOARD_PORT=6006
API_PORT=5000

.PHONY: build_dev, build_api, train, download_base_bert, .build

build_dev:
	$(MAKE) .build stage=dev tag=latest Dockerfile=Dockerfile_dev

build_api:
	$(shell echo "date: $(shell date +"%Y-%m-%d %T")" > version.yml)
	$(shell echo "sha: $(shell git rev-parse HEAD)" >> version.yml)
	$(MAKE) .build stage=api tag=latest Dockerfile=Dockerfile_api

train:
	@docker run \
		-u $(shell id -u):$(shell id -g) -t --rm\
		--name $(project)_dev \
		-e PYTHONPATH=/app/src/utils \
		-p $(TENSORBOARD_PORT):6006 \
		-v $(PWD)/:/app/ \
		bash -c "tensorboard --logdir /app/models/ && python /app/src/train.py"

start_api:
	docker run \
		-p $(API_PORT):5000 \
		-e PYTHONPATH=/app/src/utils \
		--name=$(project)_api \
		-d $(project)_api:latest

download_base_bert:
	wget -q https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip -P $(PWD)/models
	unzip -j -u $(PWD)/models/multi_cased_L-12_H-768_A-12.zip -d $(PWD)/models/pretrained/
	rm $(PWD)/models/multi_cased_L-12_H-768_A-12.zip

.build:
	docker build -t $(project)_$(stage):$(tag) -f $(Dockerfile) .

