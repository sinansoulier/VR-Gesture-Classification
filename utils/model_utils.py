from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

from sklearn.pipeline import Pipeline
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

from xgboost import XGBClassifier

class ModelUtils:
    """
    Utility class for Machine Learning models.
    """
    @staticmethod
    def save_sklearn_model_to_onnx(model, path: str, data_shape: tuple[int, int], is_xgboost=False) -> None:
        """
        Save a sklearn model to ONNX format.
        Params:
            model (sklearn.base.BaseEstimator): Model to save.
            path (str): Path to save the model.
            data_shape (tuple[int, int]): Shape of the input data.
            is_xgboost (bool): Whether the model is a XGBoost model or not.
        """
        if is_xgboost:
            pipeline = Pipeline([('classifier', model)])
            update_registered_converter(
                XGBClassifier,
                'XGBClassifier',
                calculate_linear_classifier_output_shapes,
                convert_xgboost,
                options={'nocl': [True, False], 'zipmap': [True, False]}
            )
            model = pipeline
        
        initial_type = [('float_input', FloatTensorType([1, 1, data_shape[0], data_shape[1]]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open(path, 'wb') as f:
            f.write(onx.SerializeToString())