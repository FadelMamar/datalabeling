import mlflow

def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
      exp_id = mlflow.create_experiment(name)
      return exp_id
    return exp.experiment_id


# if __name__ == '__main__':
#   import argparse
#   parser = argparse.ArgumentParser('Creates/gets an MLflow experiment and registers a detection model to the Model Registry')
#   parser.add_argument('--name', help='MLflow experiment name')
#   args = parser.parse_args()

#   mlflow.set_tracking_uri("http://localhost:5000")
#   get_experiment_id(args.name)