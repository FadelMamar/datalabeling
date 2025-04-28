from datalabeling.common.evaluation import PerformanceEvaluator
from datalabeling.common.config import EvaluationConfig
from datalabeling.ml.models import Detector

if __name__ == "__main__":
    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "all_scores"

    detector = Detector(
        path_to_weights=r"D:\datalabeling\base_models_weights\best.pt",
        confidence_threshold=0.25,
        overlap_ratio=0.2,
        tilesize=800,
        imgsz=800,
        use_sliding_window=True,
        device="cpu",
    )

    perf_eval = PerformanceEvaluator(config=eval_config)

    df_results, df_labels = perf_eval.get_preds_targets(
        images_dirs=[
            r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format\images",
        ],
        pred_results_dir=r"D:\herdnet-Det-PTR_emptyRatio_0.0\yolo_format",
        detector=detector,
        load_results=False,
    )

    metrics = perf_eval.evaluate(df_results, df_labels)
    # df = df_results.merge(df_labels,on='file_name',how='left')
