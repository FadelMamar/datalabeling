import typer

from src.common.evaluation import compute_detector_performance
from src.common.io import get_preds_targets

app = typer.Typer()


@app.command()
def main(
    data_dir: str,
    results_dir: str,
    output_report: str = "performance_report.csv",
):
    # Load predictions and ground truth
    preds, labels, img_list = get_preds_targets([data_dir], results_dir)

    # Compute overall performance
    report_df = compute_detector_performance(preds, labels, col_names=["category"])
    report_df.to_csv(output_report, index=False)
    print(f"Saved main performance report to {output_report}")


if __name__ == "__main__":
    app()
