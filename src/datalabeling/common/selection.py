import numpy as np
import pandas as pd

from .config import EvaluationConfig


# compute uncertainty
def get_uncertainty(
    df_results_per_img: pd.DataFrame,
    mode: str = "entropy",
    reoder_ascending: bool = False,
) -> pd.DataFrame:
    if mode == "entropy":
        entropy_func = lambda x: -1 * (np.log(x) * x).sum()
        df_results_per_img["uncertainty"] = df_results_per_img["all_scores"].apply(
            entropy_func
        )

    elif mode == "1-p":
        df_results_per_img["uncertainty"] = df_results_per_img["all_scores"].apply(
            lambda x: 1.0 - np.mean(x)
        )

    else:
        raise NotImplementedError("mode is not implemented yet. entropy or 1-p")

    df_results_per_img.sort_values(
        "uncertainty", axis=0, ascending=reoder_ascending, inplace=True
    )

    return df_results_per_img


# =====================
# Uncertainty Analysis
# =====================
class UncertaintyAnalyzer:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def calculate_uncertainty(self, df_results_per_img: pd.DataFrame) -> pd.DataFrame:
        """Calculate uncertainty metrics"""

        df_results_per_img = get_uncertainty(
            df_results_per_img,
            mode=self.config.uncertainty_mode,
            reoder_ascending=False,
        )

        return df_results_per_img


# =====================
# Hard Sample Analysis
# =====================
class HardSampleSelector:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.df_hard_negatives = None

    def select_hard_samples(self, df_results_per_img: pd.DataFrame) -> pd.DataFrame:
        """Identify challenging samples based on multiple criteria"""
        self._filter_by_uncertainty(df_results_per_img)
        return self.df_hard_negatives

    def save_selection_references(
        self, df_hard_negatives: pd.DataFrame, save_path_samples: str
    ) -> None:
        df_hard_negatives["image_paths"].to_csv(
            save_path_samples, index=False, header=False
        )

    def _filter_by_uncertainty(self, df_results_per_img: pd.DataFrame) -> pd.DataFrame:
        """Apply uncertainty threshold filtering"""

        df_results_per_img = get_uncertainty(
            df_results_per_img,
            mode=self.config.uncertainty_method,
            reoder_ascending=False,
        )

        mask_low_map = (df_results_per_img["map50"] < self.config.map_threshold) * (
            df_results_per_img["map75"] < self.config.map_threshold
        )

        mask_high_scores = (
            df_results_per_img[self.config.score_col] > self.config.score_threshold
        )
        mask_low_scores = df_results_per_img[self.config.score_col] < (
            1 - self.config.score_threshold
        )
        mask_selected = (
            mask_low_map * mask_high_scores
            + mask_low_map * mask_low_scores
            + (df_results_per_img["uncertainty"] > self.config.uncertainty_threshold)
        )

        self.df_hard_negatives = df_results_per_img.loc[mask_selected]
