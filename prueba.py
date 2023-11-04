import numpy as np
import pandas as pd
from baseline_model import RS_baseline_usr_mov

df_scores_train = pd.read_csv("data/scores.csv")

model = RS_baseline_usr_mov(0.5)

model.fit(df_scores_train)
print("mean_usr:", model.mean_usr)
print("mean_mov:", model.mean_mov)
print("mean:", model.mean)

preds = model.predict(
    np.array(
        [
            [1, 5],
            [1, 10],
            [43, 5],
            [42, 7],
            [1, 1],
        ]
    )
)
print("preds:", preds)
real = np.array([5, 3, 2, 1, 2])
metric = model.get_metric(real, preds)
print("Metrica:", metric)

model.save_model("models/prueba.pkl")
