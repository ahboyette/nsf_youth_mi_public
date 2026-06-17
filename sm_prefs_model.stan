
data {
  int<lower=1> N; // persons
  int<lower=1> K; // items
  array[N, K] int<lower=-1, upper=1> y; // responses: -1=missing, 0/1=observed
  int<lower=1> villages; // villages
  int<lower=1> eths; // ethnicities
  int<lower=1> ages; // ages
  array[N] int<lower=1, upper=villages> village;
  array[N] int<lower=1, upper=eths> ethnicity;
  array[N] int<lower=1, upper=ages> youth;
}

parameters {
// ---- Item parameters ----
  vector[K] b_raw; // difficulty (uncentered)
  vector[K] log_a; // log discrimination (a>0)

// ---- Random effects (non-centered) ----
  vector[villages] z_village; // village std RE
  real<lower=0> sigma_village;
  vector[eths] z_ethnicity; // ethnicity std RE
  real<lower=0> sigma_ethnicity;
  vector[ages] z_youth; // youth std RE
  real<lower=0> sigma_youth;


// ---- Person residual heterogeneity ----
  vector[N] z_person; // std normal
  real<lower=0> sigma_person;

// ---- population mean ------
  real mu_theta;

}

transformed parameters {
vector[K] b;
vector[K] a;
vector[villages] re_village;
vector[eths] re_ethnicity;
vector[ages] re_youth;
vector[N] theta;

// discrimination
a = exp(log_a);

// center on b as location anchor for identifiability: mean(b)=0
b = b_raw - rep_vector(mean(b_raw), K);

// random effects
re_village = sigma_village * z_village;
re_ethnicity = sigma_ethnicity * z_ethnicity;
re_youth = sigma_youth * z_youth;

// latent preference
  theta = mu_theta +
          re_village[village] + 
          re_ethnicity[ethnicity] + 
          re_youth[youth] + 
          sigma_person * z_person;
}

model {
// ---- Priors ----
// Items
b_raw ~ normal(0, 1);

log_a ~ normal(0.5, 0.6);

// population mean preference
mu_theta ~ normal(-1, 1);

// REs
z_village ~ std_normal();
z_ethnicity ~ std_normal();
z_youth ~ std_normal(); 
z_person ~ std_normal();

sigma_village ~ normal(0, 1.0);
sigma_ethnicity ~ normal(0, 1.0);
sigma_youth ~ normal(0, 1.0);
sigma_person ~ normal(0, 1.0); 



// ---- Likelihood (2PL logistic) ----
for (i in 1:N) {
  for (k in 1:K) {
    if(y[i, k] != -1) {   // only model observed responses
      y[i, k] ~ bernoulli_logit( a[k] * (theta[i] - b[k]) );
      }
    }
  }
}

generated quantities {
  matrix[N, K] log_lik;
  array[N, K] int y_rep;                       // posterior predictive draws for each person-item
  vector[K] item_mean = rep_vector(0, K);      // observed proportion correct
  vector[K] item_mean_rep = rep_vector(0, K);  // predicted proportion
  array[K] int item_n = rep_array(0, K);
  
  // Generate predictions and log-likelihood
  for (i in 1:N) {
    for (k in 1:K) {
      if (y[i , k] != -1) {
      // log-likelihood
      log_lik[i, k] = bernoulli_logit_lpmf(y[i, k] | a[k] * (theta[i] - b[k]));
      // posterior predictive draw
      y_rep[i, k] = bernoulli_logit_rng(a[k] * (theta[i] - b[k]));

        // accumulate for item statistics
        item_mean[k] += y[i, k];
        item_mean_rep[k] += y_rep[i, k];
        item_n[k] += 1;
      } else {
        log_lik[i, k] = 0;  // or negative_infinity()
        y_rep[i, k] = -1;   // flag as missing
      }
    }
  }
  
  // Convert to means
  for (k in 1:K) {
    if (item_n[k] > 0) {
      item_mean[k] /= item_n[k];
      item_mean_rep[k] /= item_n[k];
    }
  }
}
