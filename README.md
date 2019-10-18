# Bert in a flask
An implementation of BERT multi-class classification, which is served using a Flask API. Notice this is a rough implementation, and in no way a production/best practice setup. 
## Updates
* 25/10/2019 - Bugfixes. Code is now runnable, with a data example. Codebase still under construction, considering upgrading the project to TF 2.0
## Overview
This project should be seen as an example on how to serve BERT predictions through a flask API. The project is written to be compatible with docker version (>=19). The project uses the stackoverflow classification dataset, which can you substituted by any dataset of your choice, with minor adjustments. 
This project uses Docker and Flask to train the model and deploy it. Notice most of the building and training logic is wrapped into a `Makefile`. There are two Dockerfiles, one for training/development `Dockerfile_dev` and one for the API `Dockerfile_api`. Building these can be achieved using the following targets.
```Make
make build_dev
make build_api
```
### Training
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
make train tensorboard_port=<PORT> gpu=<GPU>
```
Tensorboard is intialized in `/apps/models/`, and the port defaults to 6006. The `gpu` parameter as an integer (defaults to 1) that specifies the gpu you want to use for training, which expects the training to happen on a single GPU.

When using your own data, make sure to change the section starting with `DATA LOADING` in `src/train.py` to your needs. 

The best model checkpoints are saved during training using `src/utils/best_checkpoints_exporter.py`, based on validation loss (can be customized). Furthermore the best model is exported at the end of training to `models/train_model/` (see `EXPORT MODEL` section in `train.py`).
### Serving
To serve prediction a minimal working example of a flask API is provided. API can be booted up using the command:
```
make start_api api_port=<PORT>
```
Once the API container is running, requests can be made in the following form, using the assigned api port:
```
curl -H "Content-Type: application/json" --request POST --data '<JSON_OBJECT>' http://localhost:<PORT>/predict
```
Where the JSON object should have the following format. Notice that it takes the input as a list, making it possible to parse a list of input bodies:
```json
{
    "X": ["Why does my code keep throwing me null pointer exception?", "I wrote a loop that i cannot exit, what am I doing wrong?"],
}
```
Notice the API does a minimum logging of each call made. The logged information can be found in `src/app.py`.
