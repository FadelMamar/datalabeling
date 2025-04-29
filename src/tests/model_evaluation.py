from datalabeling.common.evaluation import PerformanceEvaluator, HardSampleSelector
from datalabeling.common.config import EvaluationConfig
from datalabeling.ml.models import Detector


def run_perf_evaluator():
    eval_config = EvaluationConfig()
    eval_config.score_threshold = 0.25
    eval_config.map_threshold = 0.3
    eval_config.uncertainty_method = "entropy"
    eval_config.uncertainty_threshold = 4
    eval_config.score_col = "max_scores"

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

    df_metrics_per_img = perf_eval.evaluate(df_results, df_labels)

    # mining hard sampels
    sample_selector = HardSampleSelector(config=eval_config)

    df_hard_negatives = sample_selector.select_hard_samples(df_metrics_per_img)

    return df_metrics_per_img, df_hard_negatives


if __name__ == "__main__":
    pass
    # df = df_results.merge(df_labels,on='file_name',how='left')

    df_metrics_per_img, df_hard_negatives = run_perf_evaluator()

    # df_eval = metrics.merge(df_metrics_per_img_detailed,on='file_name',how='left')
    # fps = df_metrics_per_img_detailed.groupby(by=['file_name'])[['pred_FP']].sum().reset_index().sort_values('pred_FP',ascending=False)
    # fns = df_metrics_per_img_detailed.groupby(by=['file_name'])[['gt_FN']].sum().reset_index().sort_values('gt_FN',ascending=False)
    # tps = df_metrics_per_img_detailed.groupby(by=['file_name'])[['pred_TP']].sum().reset_index().sort_values('pred_TP',ascending=False)
    # tps_fps_fns = df_metrics_per_img_detailed.groupby(by=['file_name'])[['pred_TP','pred_FP','gt_FN']].sum().sort_values('pred_TP',ascending=False) * 1
    # tps_fps_fns['fp_tp_ratio'] = tps_fps_fns['pred_FP']/(tps_fps_fns['pred_TP'] + 1e-8)
    # tps_fps_fns['fn_tp_ratio'] = tps_fps_fns['gt_FN']/(tps_fps_fns['pred_TP'] + 1e-8)
