SHELL := /bin/bash
project=$(notdir $(shell pwd))
tensorboard_port=6006
api_port=5000
gpu=1

.PHONY: build_dev, build_api, train, .build

build_dev:
	$(MAKE) .build stage=dev tag=latest Dockerfile=docker/dev.Dockerfile

build_api:
	$(shell echo "date: $(shell date +"%Y-%m-%d %T")" > version.yml)
	$(shell echo "sha: $(shell git rev-parse HEAD)" >> version.yml)
	$(MAKE) .build stage=api tag=latest Dockerfile=docker/api.Dockerfile

train:
	@docker run \
		-u $(shell id -u):$(shell id -g) -it --rm\
		--name $(project)_dev \
		-e PYTHONPATH=/app/src/utils \
		-e BASE_MODEL_PREFIX=$(BASE_MODEL_PREFIX) \
		--gpus $(gpu) \
		-p $(tensorboard_port):6006 \
		-v $(PWD)/:/app/ \
		$(project)_dev:latest \
		bash -c "tensorboard --host 0.0.0.0 --logdir /app/models/trained & python /app/src/train.py"

start_api:
	@docker run \
		-p $(api_port):5000 \
		-e PYTHONPATH=/app/src/utils \
		--name=$(project)_api \
		-d $(project)_api:latest

.build:
	docker build -t $(project)_$(stage):$(tag) -f $(Dockerfile) .

