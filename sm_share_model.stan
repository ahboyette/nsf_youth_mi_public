
data {
  int<lower=1> N; // persons
  int<lower=1> K; // items
  int<lower=2> M; // number of response categories
  array[N, K] int<lower=-1, upper=M> y; // responses: -1=missing, 0/1=observed
  int<lower=1> villages; // villages
  int<lower=1> eths; // ethnicities
  int<lower=1> ages; // ages
  array[N] int<lower=1, upper=villages> village;
  array[N] int<lower=1, upper=eths> ethnicity;
  array[N] int<lower=1, upper=ages> youth;
}

parameters {
// ---- Item parameters ----
// For M cats, need M-1 thresholds per item; treating 1st and 2nd threshold differently based on assumption that most people share most things (choose 1 or 2)
  vector[K] b_raw_1;           // first threshold (1 to 2)
  vector<lower=0>[K] b_gap;    // positive gap to second threshold
  vector[K] log_a; // log discrimination (a>0)


// ---- Random effects  ----
  vector[villages] z_village; 
  real<lower=0> sigma_village;
  vector[eths] z_ethnicity; 
  real<lower=0> sigma_ethnicity;
  vector[ages] z_youth; 
  real<lower=0> sigma_youth;

// ---- Person residual heterogeneity ----
  vector[N] z_person; //
  real<lower=0> sigma_person;

// ---- population mean ------
  real mu_theta;
}

transformed parameters {
  matrix[K, 2] b;  // centered thresholds (for M=3 categories)
  vector[K] b1_centered;
  vector[K] a;
  vector[villages] re_village;
  vector[eths] re_ethnicity;
  vector[ages] re_youth;
  vector[N] theta;

// discrimination
a = exp(log_a);


// center on b1 as location anchor for identifiability: mean(b)=0
b1_centered = b_raw_1 - rep_vector(mean(b_raw_1), K);

 // Construct and center thresholds
  for (k in 1:K) {
    b[k, 1] = b1_centered[k];
    b[k, 2] = b1_centered[k] + b_gap[k];
  }


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

  // First threshold
b_raw_1 ~ normal(0, 0.5);

  // inter-threshold distance
b_gap ~ normal(1, 0.4);

  // discriminability
log_a ~ normal(0.5, 0.6); 


// Respondants

  // population mean preference
mu_theta ~ normal(0, 1);

  // REs
z_village ~ std_normal();
z_ethnicity ~ std_normal();
z_youth ~ std_normal(); 
z_person ~ std_normal();
sigma_village ~ normal(0, 0.5);
sigma_ethnicity ~ normal(0, 0.5);
sigma_youth ~ normal(0, 0.5);
sigma_person ~ normal(0, 0.5);



// ---- Likelihood ----
  for (i in 1:N) {
    for (k in 1:K) {
      if (y[i, k] != -1) {
        vector[M] probs;
        vector[M-1] cum_probs;
        
        for (m in 1:(M-1)) {
          cum_probs[m] = inv_logit(a[k] * (theta[i] - b[k, m]));
        }
        
        probs[1] = 1 - cum_probs[1];
        for (m in 2:(M-1)) {
          probs[m] = cum_probs[m-1] - cum_probs[m];
        }
        probs[M] = cum_probs[M-1];
        
        target += categorical_lpmf(y[i, k] | probs);
      }
    }
  }
}

generated quantities {
  matrix[N, K] log_lik;
  array[N, K] int y_rep;  // posterior predictive draws for each person-item
  matrix[K, M] item_prop = rep_matrix(0, K, M); // proportion in each category
  matrix[K, M] item_prop_rep = rep_matrix(0, K, M);
  array[K] int item_n = rep_array(0, K);
  
    for (i in 1:N) {
    for (k in 1:K) {
      if (y[i, k] != -1) {
        // Compute probabilities for this response
        vector[M] probs;
        vector[M-1] cum_probs;
        
        for (m in 1:(M-1)) {
          cum_probs[m] = inv_logit(a[k] * (theta[i] - b[k, m]));
        }
        
        probs[1] = 1 - cum_probs[1]; // P(Y = 1)
        for (m in 2:(M-1)) {
          probs[m] = cum_probs[m-1] - cum_probs[m]; // P(Y = m)
        }
        probs[M] = cum_probs[M-1]; // P(Y = M)
        
        // Log-likelihood
        log_lik[i, k] = categorical_lpmf(y[i, k] | probs);
        
        // Posterior predictive draw
        y_rep[i, k] = categorical_rng(probs);
        
        // Accumulate proportions
        item_prop[k, y[i, k]] += 1;
        item_prop_rep[k, y_rep[i, k]] += 1;
        item_n[k] += 1;
      } else {
        log_lik[i, k] = 0;
        y_rep[i, k] = -1;
      }
    }
  }
  
  // Convert to proportions
  for (k in 1:K) {
    if (item_n[k] > 0) {
      for (m in 1:M) {
        item_prop[k, m] /= item_n[k];
        item_prop_rep[k, m] /= item_n[k];
      }
    }
  }
}
