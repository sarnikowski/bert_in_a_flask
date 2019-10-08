## Under construction

### TODO
- Decide on either Makefile or compose logic
- Add data to example (stackoverflow)
- Fix serving_input_receiver_fn
- Add examples on using different performance metrics
- Make necessary changes to move from nvidia-docker2 to docker 19.

### BERT in a flask
An implementation of BERT multi-class classification, which is served using a Flask API. Notice this is a rough implementation, and in no way a production/best practice setup.

### Overview
This project should be seen as a starting point, which can be customized to specific needs. To run this project you need to have docker version (>=19). The project uses the stackoverflow classification dataset, which can you substituted by any dataset of your choice, with minor adjustments. 
This project uses Docker and Flask to train the model and deploy it. Notice most of the building and training logic is wrapped into a `Makefile`. There are two Dockerfiles, one for training/development `Dockerfile_dev` and one for the API `Dockerfile_api`. Building these can be achieved using the following targets.
```Make
make build_dev
make build_api
```
I realize a lot of the Makefile logic could be contained in a docker-compose.yml file. That might change in the future.
### Training
To train the model first download the base model.
```
make download_base_bert
``` 
(notice the base-model link in the Makefile refers to the model `BERT-Base, Multilingual Cased (New, recommended)`, change the link if you want to use a different base model). 
To train the model run:
```
make train
```
When using your own data, make sure to change the section starting with `DATA LOADING` in `src/train.py` to your needs. 

The best model checkpoints are saved during training using `src/utils/best_checkpoints_exporter.py`, based on validation loss (can be customized). Furthermore the best model is exported at the end of training to `models/train_model/` (see `EXPORT MODEL` section in `train.py`).
### Serving
To serve prediction a minimal working example of a flask API is provided. API can be booted up using the command (runs on port 5000, change if necessary):
```
make start_api
```
Once the API container is running, requests can be made in the following form:
```
curl -H "Content-Type: application/json" --request POST --data '<JSON_OBJECT>' http://localhost:5000/predict
```
Where the JSON object should have the following format:
```json
{
    "body": "<BODY>",
}
```

