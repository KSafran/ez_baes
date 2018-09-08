import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pystan
import pickle
from huey import RedisHuey

huey = RedisHuey('iris')

def get_iris_dataset():
    iris_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    return pd.read_csv(iris_file,
                            names=['sepal_length',
                                   'sepal_width',
                                   'petal_length',
                                   'petal_width',
                                   'species'])

def format_stan_data(iris_data):
    # Simple multi-level model, predict petal length using sepal length
    # and random effects from each species.
    species = LabelEncoder()
    iris_data['species_num'] = species.fit_transform(iris_data['species']) + 1

    stan_iris_data = {
        'n': iris_data.shape[0],
        'n_species': max(iris_data['species_num']),
        'sepal_length': iris_data['sepal_length'],
        'species': iris_data['species_num'],
        'petal_length': iris_data['petal_length']
    }
    return stan_iris_data

def train_iris_model(data, output_file):
    iris_model = pystan.StanModel(file='iris.stan')
    iris_fit = iris_model.sampling(data=data, iter=1000,
                              chains=4)

    # you must pickle the stan model before the stan fit
    with open(output_file, 'wb') as f:
        pickle.dump({'model':iris_model,
                     'fit':iris_fit}, f)
    return iris_fit

@huey.task()
def fit_iris():
    iris_data = get_iris_dataset()
    stan_iris_data = format_stan_data(iris_data)
    train_iris_model(stan_iris_data, '../tmp/iris.pkl')
    return True

if __name__ == '__main__':
    fit_iris()

