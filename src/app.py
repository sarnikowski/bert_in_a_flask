import logging
import sys
import yaml
import json

from flask import Flask, request, g, make_response
from predict import Predict
from time import time

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s in %(name)s: %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = Flask("app")

with open("/app/version.yml") as stream:
    version = yaml.safe_load(stream)

predict = Predict()


@app.route("/predict", methods=["POST"])
def handle_request():
    data = request.get_json(force=True)
    output = predict.predict(data)
    output["model"] = "BERT {} {}".format(version["date"], version["sha"][:8])
    return make_response(output)


@app.route("/health_check", methods=["GET"])
def health_check():
    return make_response({"text": "I am alive"})


@app.before_request
def before_request():
    g.time = time()
    g.user_ip = request.headers.get("X-Forwarded-For")
    g.user_agent = request.user_agent.string


@app.after_request
def after_request(response):
    params = {
        "endpoint": request.path,
        "response": response.get_json(force=True, silent=True),
        "status_code": response.status_code,
        "response_time": time() - g.time,
        "user_ip": g.user_ip,
        "user_agent": g.user_agent,
    }
    app.logger.info("Request log: {}".format(json.dumps(params)))
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
