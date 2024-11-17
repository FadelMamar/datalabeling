from dotenv import load_dotenv
load_dotenv("../.env")

from datalabeling.dataset.sampling import (compute_predictions, 
                                            load_prediciton_results,
                                            load_groundtruth, 
                                            compute_detector_performance,
                                            select_hard_samples)



if __name__ == "__main__":

    # compute predictions
    path_to_results = compute_predictions(weight_path=r"C:\Users\FADELCO\OneDrive\Bureau\datalabeling\models\best_openvino_model",
                                        task='obb',
                                        iou=0.6,
                                        half=True,
                                        imgsz=1280,
                                        results_dir_name=r"D:\validation_results",
                                        val_run_name='hard_samples',
                                        data_config_yaml=r"C:\Users\FADELCO\OneDrive\Bureau\datalabeling\data\data_config.yaml",
                                        batch_size=1)

    # load prediction results
    df_results = load_prediciton_results(path_to_results)

    # load gt
    df_labels, col_names = load_groundtruth(data_config_yaml=r"C:\Users\FADELCO\OneDrive\Bureau\datalabeling\data\data_config.yaml")

    # compute mAP
    df_results_per_img = compute_detector_performance(df_results=df_results,
                                                    df_labels=df_labels,
                                                    col_names=col_names)

    # save hard samples
    df_hard_negatives = select_hard_samples(df_results_per_img=df_results_per_img,
                                            map_thrs=0.3,
                                            score_thrs=0.7,
                                            save_path_samples=r"D:\hard_negative_samples\hard_samples.txt",
                                            root='D:\\',
                                            save_data_yaml=r"C:\Users\FADELCO\OneDrive\Bureau\datalabeling\data\hard_samples.yaml"
                                            )