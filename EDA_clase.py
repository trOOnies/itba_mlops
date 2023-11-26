import os
import pandas as pd
import mlflow
from matplotlib import pyplot as plt

mlflow.set_experiment("EDA")

PREFIX = "raw"

df_scores = pd.read_csv(os.path.join(PREFIX, "scores_0.csv"))
df_peliculas = pd.read_csv(os.path.join(PREFIX, "peliculas_0.csv"))
df_personas = pd.read_csv(os.path.join(PREFIX, "personas_0.csv"))
df_trabajadores = pd.read_csv(os.path.join(PREFIX, "trabajadores_0.csv"))
df_usuarios = pd.read_csv(os.path.join(PREFIX, "usuarios_0.csv"))

df_personas = df_personas.astype({"year of birth": int})

# mlflow.log_artifacts("data")

# Grafico el histograma de scores
fig = plt.figure(figsize=(9,3))
plt.hist(df_scores.rating, bins=5)
plt.title(f"Histograma de los ratings. Promedio = {df_scores.rating.mean()}")
mlflow.log_figure(fig, "Histograma ratings.png")

# Voy a buscar correlación entre fecha de nacimiento/género y score
df_merge = df_scores.merge(df_personas, left_on="user_id", right_on="id")
df_mean = df_merge.groupby(["year of birth", "Gender"]).rating.mean().reset_index()
fig = plt.figure(figsize=(9, 3))
df_mean.query("Gender == 'M'").set_index("year of birth").rating.plot(label="Male")
df_mean.query("Gender == 'F'").set_index("year of birth").rating.plot(label="Female")
plt.legend()
plt.ylabel("Rating")
mlflow.log_figure(fig, "Rating promedio por año.png")

# Guardo como métrica el score promedio
aux = df_scores.rating.describe()
mlflow.log_metric("Avg Score", aux["mean"])
mlflow.log_metric("Min Score", aux["min"])
mlflow.log_metric("Max Score", aux["max"])
mlflow.log_metric("Score Std", aux["std"])
mlflow.log_metric(
    "Sparcity",
    df_scores.shape[0] / (df_personas.shape[0] * df_peliculas.shape[0])
)

mlflow.end_run()
