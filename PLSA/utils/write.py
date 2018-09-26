from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline
from xgboost.sklearn import XGBClassifier
import json

def xgboost_to_pmml(data_X, data_y, par_file, save_model_as):
    """Save Xgboost Model to PMMl file.

    Parameters
    ----------
    data_X : pandas.DataFrame
        Variables of train data.
    date_y : pandas.DataFrame
        Lables of train data.
    par_file : str
        File path of model's parameters.
    save_model_as : str
        File path of PMML.

    Returns
    -------
    None
        Generate PMML file locally as `save_model_as` given.

    Examples
    --------
    >>> xgboost_to_pmml(data_x, data_y, "par.json", "model.pmml")
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
    pipeline.fit(data_X, data_y)
    # Save Model
    sklearn2pmml(pipeline, save_model_as, with_repr=True)