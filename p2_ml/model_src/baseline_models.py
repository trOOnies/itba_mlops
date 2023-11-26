import numpy as np
import pickle
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from pandas import DataFrame


class RS_baseline_usr_mov:
    def __init__(self, p) -> None:
        self.p = p

    def fit(self, df_scores_train: "DataFrame") -> None:
        df_scores_train = df_scores_train.astype({"user_id": int, "movie_id": int})
        self.mean_usr = df_scores_train.groupby("user_id").rating.mean()
        self.mean_mov = df_scores_train.groupby("movie_id").rating.mean()
        self.mean = df_scores_train.rating.mean()

    def get_usr_mov(self, row) -> Tuple[float, float]:
        if row[0] in self.mean_usr:
            usr = self.mean_usr[row[0]]
        else:
            usr = self.mean
        if row[1] in self.mean_mov:
            mov = self.mean_mov[row[1]]
        else:
            mov = self.mean
        return usr, mov

    def predict(self, X) -> np.ndarray:
        # usr, mov
        scores = np.array([self.get_usr_mov(row) for row in X], dtype=float)
        assert scores.shape == (X.shape[0], 2)
        return self.p * scores[:, 0] + (1 - self.p) * scores[:, 1]

    @staticmethod
    def get_metric(y, y_pred) -> float:
        return np.sqrt((y-y_pred)**2).mean()

    def score(self, X, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return self.get_metric(y, y_pred)

    def save_model(self, filename) -> None:
        with open(filename, "wb") as f:
            pickle.dump([self.mean_usr, self.mean_mov, self.mean, self.p], f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, "rb") as f:
            mean_usr, mean_mov, mean, p = pickle.load(f)
        model = RS_baseline_usr_mov(p)
        model.mean_usr = mean_usr
        model.mean_mov = mean_mov
        model.mean = mean
        model.p = p
        return model


class RS_baseline:
    def fit(self, df_scores_train: "DataFrame") -> None:
        df_scores_train = df_scores_train.astype({"user_id": int, "movie_id": int})
        self.mean_usr = df_scores_train.groupby("user_id").rating.mean()
        self.mean_mov = df_scores_train.groupby("movie_id").rating.mean()
        self.mean = df_scores_train.rating.mean()

    def predict(self, X) -> np.ndarray:
        # usr, mov
        return np.fromiter((self.mean for _ in X), dtype=float)

    def score(self, X, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.sqrt((y-y_pred)**2).mean()

    def save_model(self, filename) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.mean, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, "rb") as f:
            mean, p = pickle.load(f)
        model = RS_baseline(p)
        model.mean_usr = mean
        model.mean_mov = mean
        model.mean = mean
        model.p = p
        return model
