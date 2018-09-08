data{
    int<lower=0> n;
    int<lower=0> n_species;
    vector[n] sepal_length;
    int<lower=1, upper=n_species> species[n];
    vector[n] petal_length;
}
parameters {
    real base_b0;
    vector[n_species] b_0_species;
    real base_b_sepal;
    vector[n_species] b_sepal_species;
    real<lower=0> sigma_y;
}
transformed parameters {
    vector[n] yhat;
    for (i in 1:n)
        yhat[i] = b_0_species[species[i]] + b_sepal_species[species[i]] * sepal_length[i];
}
model {
    b_0_species ~ normal(base_b0, 0.1);
    b_sepal_species ~ normal(base_b_sepal, 0.1);
    sigma_y ~ normal(0, 10);

    petal_length ~ normal(yhat, sigma_y);
}
