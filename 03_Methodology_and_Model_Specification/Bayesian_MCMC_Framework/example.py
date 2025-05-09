import pymc as pm

with pm.Model() as hierarchical_spike_slab_model:
    # Hyperpriors for themes
    mu_theme = pm.Normal('mu_theme', mu=0, sigma=1, shape=num_themes)
    tau_theme = pm.HalfCauchy('tau_theme', beta=1, shape=num_themes)

    # Spike-and-Slab setup
    slab_sd = pm.HalfCauchy('slab_sd', beta=1)
    spike_sd = 0.01

    # Prior inclusion probabilities informed by Jensen et al. (2023)
    pi_theme = pm.Beta('pi_theme', alpha=2, beta=1, shape=num_themes)

    gamma = pm.Bernoulli('gamma', p=pi_theme[theme_index], shape=num_factors)

    beta = pm.Normal('beta', mu=mu_theme[theme_index],
                     sigma=gamma * slab_sd + (1 - gamma) * spike_sd,
                     shape=num_factors)

    # Likelihood
    sigma_obs = pm.InverseGamma('sigma_obs', alpha=2, beta=2)
    mu_obs = pm.math.dot(X, beta)
    returns_obs = pm.Normal('returns_obs', mu=mu_obs, sigma=sigma_obs, observed=r)

    trace = pm.sample(2000, tune=1000, target_accept=0.9)