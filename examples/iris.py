import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pystan

iris_file = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

iris_data = pd.read_csv(iris_file,
                        names=['sepal_length',
                               'sepal_width',
                               'petal_length',
                               'petal_width',
                               'species'])

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

iris_model = pystan.StanModel(file='examples/iris.stan')
iris_fit = iris_model.sampling(data=stan_iris_data, iter=1000,
                          chains=4)
