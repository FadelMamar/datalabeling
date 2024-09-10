import mlflow

def get_experiment_id(name:str):
  """Gets mlflow experiments id

  Args:
      name (str): mlflow experiment name

  Returns:
      str: experiment id
  """
  exp = mlflow.get_experiment_by_name(name)
  if exp is None:
    exp_id = mlflow.create_experiment(name)
    return exp_id
  return exp.experiment_id
