from .predictor import MyModelAPI, DetectionService
import litserve as ls
import os
import ray
from ray import serve


def run_inference_server(port=4141):
    api = MyModelAPI()
    server = ls.LitServer(
        api,
        max_batch_size=1,
    )
    server.run(port=port)
