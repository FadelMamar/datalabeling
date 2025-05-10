from .predictor import MyModelAPI
import litserve as ls


def run_inference_server(port=4141):
    api = MyModelAPI()
    server = ls.LitServer(
        api,
        max_batch_size=1,
    )
    server.run(port=port)
