# Bert in a flask
This repo contains an implementation of [Bert TF 2.0](https://github.com/kpe/bert-for-tf2) multi-class classification, served using a Flask API. The purpose of the project is to show how to serve BERT predicitions through as simple Flask API.

The code is compatible with [Tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf), and [Docker>=19](https://docs.docker.com/). The project uses the [Stackoverflow dataset](https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv).
## NEWS
* **06-01/2020** - Codebase has been overhauled and ported to [Tensorflow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf). The implementation is now based on [Bert TF 2.0](https://github.com/kpe/bert-for-tf2). This has simplified the code base immensely. 
* **25/10/2019** - Bugfixes. Code is now runnable, with a data example. Codebase still under construction, considering upgrading the project to TF 2.0
## USAGE
All logic for running the code is wrapped into a `Makefile` for convenience. There are two Dockerfiles, one for training: `Dockerfile_dev` and one for the API: `Dockerfile_api`. Building these can be achieved using the following targets.
```Make
make build_dev
make build_api
```
Notice you should rerun `make build_api` whenever you make changes to the serving code.
### TRAINING
For model training, this repo uses a [google base model](https://github.com/google-research/bert). Prior to training, download this using the following command:
```
make download_base_bert
``` 
The base-model link in the Makefile refers to the model `BERT-Base, Multilingual Cased (New, recommended)`. If you wish to use a different google model change the variable `BASE_MODEL_PREFIX`. 

Next download the stack [Stackoverflow dataset](https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv).

```
make download_data
```
Depending on your GPU hardware, you might need to adjust either the `max_seq_len` and/or `batch_size` in `src/train.py` to avoid out-of-memory exceptions during training.

To train the model run the following command:
```
make train TENSORBOARD_PORT=<PORT> GPU=<GPU>
```
Tensorboard is intialized in `models/`, and the port defaults to 6006. The `GPU` parameter as an integer (defaults to 1) specifies the GPU you want to use for training. This code is written for single GPU setups.

Model files and tensorboard logs are outputted to a datetime stamped folder under `models/`.
### INFERENCE
To serve predictions, a minimal working example of a flask API is provided. 
The API loads the latest trained model, by looking in the file `models/latest_model_config.json`, which is overwritten everytime a model is trained. Modify this file, if you wish to use a different model. 

API can be booted up using the command:
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
    "x": ["Should i use public or private when declaring a variable in a class?", "I    get ImportError everytime i try to import a module in my  main.py script"]
}
```
The API does a minimum logging of each call made. The logged information can be found in `src/app.py`. Below is an example of a request log:
```json
{
    "endpoint": "/predict", 
    "response": {   
        "model": "BERT 2020-01-06 22:06:23 4b4e44b2", 
        "predictions": ["java", "python"], 
        "probabilities": [0.9852411150932312, 0.9999822378158569]
    }, 
    "status_code": 200, 
    "response_time": 0.48097777366638184, 
    "user_ip": null, 
    "user_agent": "curl/7.58.0"
}
```
## LICENSE
MIT ([License file](LICENSE))