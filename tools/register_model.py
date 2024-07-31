import argparse
import cloudpickle
import mlflow
from sys import version_info
from datalabeling.mlflow import get_experiment_id
from datalabeling.mlflow import  model_wrapper
from datalabeling.mlflow import DetectorWrapper

# set local tracking server
TRACKING_URI="http://localhost:5000"
mlflow.set_tracking_uri(TRACKING_URI)

PYTHON_VERSION = "{major}.{minor}.1".format(major=version_info.major,
                                            minor=version_info.minor)

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python~={}'.format(PYTHON_VERSION),
        'pip',
          {
            'pip': [
                'mlflow',
                'pillow',
                'ultralytics',
                'sahi',
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'torch>=2.0.0'
            ],
          },
    ],
    'name': 'wildai_env'
}

def main():
    parser = argparse.ArgumentParser('Creates/gets an MLflow experiment and registers a detection model to the Model Registry')
    parser.add_argument('--name', help='MLflow experiment name')
    parser.add_argument('--model', help='Path to saved PyTorch model')
    parser.add_argument('--model-name', help='Registered model name')

    args = parser.parse_args()

    artifacts = {'path': args.model}

    model = DetectorWrapper()

    exp_id = get_experiment_id(args.name)

    cloudpickle.register_pickle_by_value(model_wrapper)

    with mlflow.start_run(experiment_id=exp_id):
        mlflow.pyfunc.log_model(
            'finetuned',
            python_model=model,
            conda_env=conda_env,
            artifacts=artifacts,
            registered_model_name=args.model_name
        )


if __name__ == '__main__':
    main()
