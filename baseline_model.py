import mlflow
from model_src import RS_baseline_usr_mov


class MLF_RS_baseline_usr_mov(mlflow.pyfunc.PythonModel):
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

    def load_context(self, context=None):        
        self.model = RS_baseline_usr_mov.load_model(self.model_checkpoint)

    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        return predictions
