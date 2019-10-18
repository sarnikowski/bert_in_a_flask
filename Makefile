SHELL := /bin/bash
project=$(notdir $(shell pwd))
tensorboard_port=6006
api_port=5000
gpu=1

.PHONY: build_dev, build_api, train, download_base_bert, download_data, .build

build_dev:
	$(MAKE) .build stage=dev tag=latest Dockerfile=Dockerfile_dev

build_api:
	$(shell echo "date: $(shell date +"%Y-%m-%d %T")" > version.yml)
	$(shell echo "sha: $(shell git rev-parse HEAD)" >> version.yml)
	$(MAKE) .build stage=api tag=latest Dockerfile=Dockerfile_api

train:
	@docker run \
		-u $(shell id -u):$(shell id -g) -it --rm\
		--name $(project)_dev \
		-e PYTHONPATH=/app/src/utils \
		--gpus $(gpu) \
		-p $(tensorboard_port):6006 \
		-v $(PWD)/:/app/ \
		$(project)_dev:latest \
		bash -c "tensorboard --logdir /app/models/ & python /app/src/train.py"

start_api:
	@docker run \
		-p $(api_port):5000 \
		-e PYTHONPATH=/app/src/utils \
		--name=$(project)_api \
		-d $(project)_api:latest

download_base_bert:
	wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip -P $(PWD)/models
	unzip -j -u $(PWD)/models/multi_cased_L-12_H-768_A-12.zip -d $(PWD)/models/pretrained/
	rm $(PWD)/models/multi_cased_L-12_H-768_A-12.zip

download_data:
	wget https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv -P data/

.build:
	docker build -t $(project)_$(stage):$(tag) -f $(Dockerfile) .

