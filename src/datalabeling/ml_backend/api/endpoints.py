def run_inference_server(port=4141):
    import litserve as ls

    from ..services.predictor import MyModelAPI

    api = MyModelAPI()
    server = ls.LitServer(
        api,
        max_batch_size=1,
    )
    server.run(port=port)
