from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline
from xgboost.sklearn import XGBClassifier
import json

def xgboost_to_pmml(data_X, date_y, par_file, save_model_as):
    """
    Save Xgboost Model to PMMl file.

    Parameters:
        data_X, date_y: Train data.
        par_file: File path of model's parameters.
        save_model_as: File path of PMML.

    Returns:

    Examples:
        xgboost_to_pmml(data_x, data_y, "par.json", "model.pmml")
    """
    # Create Xgboost Model
    with open(par_file, "r") as f:
        par = json.load(f)
    xgb_now = XGBClassifier(**par)
    # Create Pipeline
    pipeline = PMMLPipeline([
        ("classifier", xgb_now)
    ])
    # Fit Model
    pipeline.fit(data_X, date_y)
    # Save Model
    sklearn2pmml(pipeline, save_model_as, with_repr=True)