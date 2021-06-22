# imports

#from numpy.lib.twodim_base import tril_indices
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class Trainer(TimeFeaturesEncoder, DistanceTransformer):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe

    
    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    from TaxiFareModel.data import get_data, clean_data
    from sklearn.model_selection import train_test_split
 
    print('entering encoder tests')
    print('getting and cleaning data')
    df = clean_data(get_data(10))
    # prepare X and y
    print('Setting X and y')
    y = df.pop("fare_amount")
    X = df
    print(type(y))
    print(type(X))
    print('Instanciate trainer:')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, y_train)
#    print(trainer.pipeline)
#    trainer.set_pipeline()
#    print(trainer.pipeline)
    trainer.run()
    print(trainer.evaluate(X_train, y_train))
    print(trainer.evaluate(X_test, y_test))
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
