from datalabeling.ml_backend.api.endpoints import run_inference_server

if __name__ == "__main__":
    # detector = Detector(path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
    #                     confidence_threshold=0.25,
    #                     overlap_ratio=0.2,
    #                     tilesize=800,
    #                     imgsz=800,
    #                     use_sliding_window=True,
    #                     device='cpu'
    #                     )

    # img_path = r"D:\paul_data\DJI_20231002100957_0001.JPG"

    # result = detector.predict_directory(images_paths=[img_path,],
    #                                      return_coco=True,
    #                                      return_gps=True,
    #                                      as_dataframe=True,
    #                                      )

    run_inference_server(port=4141)
