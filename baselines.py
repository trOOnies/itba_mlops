import pandas as pd
import mlflow
import argparse
from baseline_model import RS_baseline_usr_mov

mlflow.set_experiment("baselines")

parser = argparse.ArgumentParser()
parser.add_argument("-p", dest="p", required=True,
                    type=float, help="Ponderacion del usuario por sobre la pelicula")
args = parser.parse_args()

# ----

df_scores_train = pd.read_csv("data/scores.csv")

model = RS_baseline_usr_mov(args.p)
model.fit(df_scores_train)

preds = model.predict(
    df_scores_train[["user_id", "movie_id"]].values
)
metric = model.get_metric(df_scores_train.rating.values, preds)
mlflow.log_metric("RMSE", metric)

dst_path = "models/prueba_" + "{:.6f}".format(args.p) + ".pkl"
model.save_model(dst_path)
mlflow.log_artifact(dst_path)
