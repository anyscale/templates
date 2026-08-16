import time
from typing import Dict, Any

import ray
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# states to inspect
STATES = ["INITIALIZED", "RUNNING", "DONE"]

DECISION_TREE_CONFIGS = {"max_depth": 10, "name": "decision_tree"}

RANDOM_FOREST_CONFIGS = {"n_estimators": 25, "name": "random_forest"}

XGBOOST_CONFIGS = {
    "max_depth": 10,
    "n_estimators": 25,
    "lr": 0.1,
    "eta": 0.3,
    "colsample_bytree": 1,
    "name": "xgboost",
}


def load_california_housing():
    """
    Fetch the California housing dataset as (features, target) frames.

    Call this once, on the driver, and hand the frames to the actors through
    Ray's object store. Do not call it from actor code: sklearn downloads into a
    single unlocked path under ``~/scikit_learn_data`` and deletes the archive
    when it is done, so two actor processes starting together on the same node
    truncate or unlink each other's copy of ``cal_housing.tgz``.
    """
    return fetch_california_housing(return_X_y=True, as_frame=True)


class ActorCls:
    """
    Base class for our Ray Actor workers models
    """

    def __init__(self, configs: Dict[Any, Any], X_data, y_data) -> None:
        self.configs = configs
        self.name = configs["name"]
        self.state = STATES[0]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=4
        )

        self.model = None

    def get_name(self) -> str:
        return self.name

    def get_state(self) -> str:
        return self.state

    def train_and_evaluate_model(self) -> Dict[Any, Any]:
        """
        Overwrite this function in super class
        """
        pass


@ray.remote
class RFRActor(ActorCls):
    """
    An actor model to train and score the calfornia house data using Random Forest Regressor
    """

    def __init__(self, configs, X_data, y_data):
        super().__init__(configs, X_data, y_data)
        self.estimators = configs["n_estimators"]

    def train_and_evaluate_model(self) -> Dict[Any, Any]:
        """
        Train the model and evaluate and report MSE
        """

        self.model = RandomForestRegressor(
            n_estimators=self.estimators, random_state=42
        )

        print(
            f"Start training model {self.name} with estimators: {self.estimators} ..."
        )

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.state = STATES[1]
        y_pred = self.model.predict(self.X_test)
        score = mean_squared_error(self.y_test, y_pred)
        self.state = STATES[2]

        end_time = time.time()
        print(
            f"End training model {self.name} with estimators: {self.estimators} took: {end_time - start_time:.2f} seconds"
        )

        return {
            "state": self.get_state(),
            "name": self.get_name(),
            "estimators": self.estimators,
            "mse": round(score, 4),
            "time": round(end_time - start_time, 2),
        }


@ray.remote
class DTActor(ActorCls):
    """
    An actor model to train and score the calfornia house data using Decision Tree Regressor
    """

    def __init__(self, configs, X_data, y_data):
        super().__init__(configs, X_data, y_data)
        self.max_depth = configs["max_depth"]

    def train_and_evaluate_model(self) -> Dict[Any, Any]:
        """
        Train the model and evaluate and report MSE
        """

        self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)

        print(
            f"Start training model {self.name} with max depth: { self.max_depth } ..."
        )

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.state = STATES[1]
        y_pred = self.model.predict(self.X_test)
        score = mean_squared_error(self.y_test, y_pred)
        self.state = STATES[2]

        end_time = time.time()
        print(
            f"End training model {self.name} with max_depth tree: {self.max_depth} took: {end_time - start_time:.2f} seconds"
        )

        return {
            "state": self.get_state(),
            "name": self.get_name(),
            "max_depth": self.max_depth,
            "mse": round(score, 4),
            "time": round(end_time - start_time, 2),
        }


@ray.remote
class XGBoostActor(ActorCls):
    """
    An actor model to train and score the calfornia house data using XGBoost Regressor
    """

    def __init__(self, configs, X_data, y_data):
        super().__init__(configs, X_data, y_data)

        self.max_depth = configs["max_depth"]
        self.estimators = configs["n_estimators"]
        self.colsample = configs["colsample_bytree"]
        self.eta = configs["eta"]
        self.lr = configs["lr"]

    def train_and_evaluate_model(self) -> Dict[Any, Any]:
        """
        Train the model and evaluate and report MSE
        """

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=self.colsample,
            eta=self.eta,
            learning_rate=self.lr,
            max_depth=self.max_depth,
            n_estimators=self.estimators,
            random_state=42,
        )

        print(
            f"Start training model {self.name} with estimators: {self.estimators} and max depth: { self.max_depth } ..."
        )
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.state = STATES[1]
        y_pred = self.model.predict(self.X_test)
        score = mean_squared_error(self.y_test, y_pred)
        self.state = STATES[2]

        end_time = time.time()
        print(
            f"End training model {self.name} with estimators: {self.estimators} and max depth: { self.max_depth } and took: {end_time - start_time:.2f}"
        )

        return {
            "state": self.get_state(),
            "name": self.get_name(),
            "max_depth": self.max_depth,
            "mse": round(score, 4),
            "estimators": self.estimators,
            "time": round(end_time - start_time, 2),
        }
