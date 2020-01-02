# Bert in a flask
This repo contains an implementation of [Bert TF 2.0](https://github.com/kpe/bert-for-tf2) multi-class classification, served using a Flask API. The purpose of the project is to show how to serve BERT predicitions through as simple Flask API.

The code is compatible with [Tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf), and [Docker>=19](https://docs.docker.com/). The project uses the [Stackoverflow dataset](https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv).
## UPDATES
* 25/10/2019 - Bugfixes. Code is now runnable, with a data example. Codebase still under construction, considering upgrading the project to TF 2.0
## USAGE
All logic for running the code is wrapped into a `Makefile` for convenience. There are two Dockerfiles, one for training `Dockerfile_dev` and one for the API `Dockerfile_api`. Building these can be achieved using the following targets.
```Make
make build_dev
make build_api
```
Notice you should rerun `make build_api` if you have trained a new model that you want to serve.
### TRAINING
To train the model first download the base model.
```
make download_base_bert
``` 
The base-model link in the Makefile refers to the model `BERT-Base, Multilingual Cased (New, recommended)`, change the link if you want to use a different base model. 

Next download the stack overflow data set:
```
make download_data
```
To train the model run the following command.:
```
make train TENSORBOARD_PORT=<PORT> GPU=<GPU>
```
Tensorboard is intialized in `/apps/models/`, and the port defaults to 6006. The `GPU` parameter as an integer (defaults to 1) that specifies the GPU you want to use for training. Notice this code is written for single GPU setups.

The best model checkpoints are saved during training using `src/utils/best_checkpoints_exporter.py`, based on validation loss (can be customized). Furthermore the best model is exported at the end of training to `models/train_model/` (see `EXPORT MODEL` section in `train.py`).
### INFERENCE
To serve predictions, a minimal working example of a flask API is provided. API can be booted up using the command:
```
make start_api api_port=<PORT>
```
Once the API container is running, requests can be made in the following form, using the assigned api port:
```
curl -H "Content-Type: application/json" --request POST --data '<JSON_OBJECT>' http://localhost:<PORT>/predict
```
The JSON object should have the following format. Notice that it takes the input as a list, making it possible to parse a list of input bodies:
```json
{
    "X": ["Why does my code keep throwing me null pointer exception?", "I wrote a loop that i cannot exit, what am I doing wrong?"],
}
```
The API does a minimum logging of each call made. The logged information can be found in `src/app.py`.
## LICENSE
MIT ([License file](LICENSE))