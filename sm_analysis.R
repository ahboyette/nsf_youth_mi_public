# packages ####
library(cmdstanr)
library(posterior)
library(tidybayes)
library(rethinking)
library(dplyr)
library(tidyr)
library(stringr)
library(readr)
library(ggplot2)
library(ggdist)
library(patchwork)
library(ggridges)
library(viridis)


# MARKET PREFERENCES MODEL ####

  #### Stan code of Market Preferences Model as string ####
pref_stan_code <- "
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
  vector[villages] z_village; 
  real<lower=0> sigma_village;
  vector[eths] z_ethnicity; 
  real<lower=0> sigma_ethnicity;
  vector[ages] z_youth;
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
"

# create temporary file path to run Stan model
tmp_stan_path_pref <- tempfile(fileext = ".stan")
writeLines(pref_stan_code, tmp_stan_path_pref)


  #### Run Market Preferences Stan model ####
  
# load data
p <- read_csv("https://raw.githubusercontent.com/ahboyette/nsf_youth_mi_public/main/sm_pref_data.csv")

# remove all rows with missing values for item before pivoting
nrow(p) # 3900
p <- p %>%
  filter(!is.na(item)) %>%      # remove NA items
  distinct(id, item, .keep_all = TRUE)   # remove duplicates
nrow(p) # should remove 49 missing values leaving 3851 rows


# convert to wide N x K matrix
y <- p %>%
  select(id, item, y) %>%
  pivot_wider(
    names_from = item,
    values_from = y) %>%
  arrange(id) %>%
  select(-id) %>%          # remove id column for Stan
  select(order(as.numeric(names(.)))) %>% # have the columns in order for item_labels
  as.matrix()


# fill in missing values
y[is.na(y)] <- -1   # -1 represents missing values
table(y)
# y
#   -1    0    1 
#   49 2350 1516 


# Verify there are 15 columns for each of the 15 items
colnames(y) # should be 15


# describe matrix for Stan
N <- nrow(y)  # number of persons
K <- ncol(y)  # number of items


# Prepare per-person demographics
people <- p %>%
  distinct(id, .keep_all = TRUE) %>%  # one row per person
  arrange(id) %>% 
  mutate(
    ethnicity_f = factor(ethnicity, levels = c(1,2), labels = c("BaYaka", "Bantu")),
    youth_f = factor(youth, levels = c(1,2), labels = c("Adult", "Adolescent")),
    village_f = factor(village)#,
    #    sex_f = factor(sex)
  )


# Convert factors to integers for Stan
ethnicity <- as.integer(people$ethnicity_f)     # 1 = BaYaka, 2 = Bantu
youth <- as.integer(people$youth_f)             # 1 = Adult, 2 = Youth
village <- as.integer(people$village_f)


# Determine group counts
villages <- max(village)
eths     <- max(ethnicity)
ages     <- max(youth)


# Assemble Stan data list
stan_data_pref <- list(
  y = y,             # N x K response matrix with -1 for missing
  N = N,             
  K = K,             
  villages = villages,
  eths = eths,
  ages = ages,
  village = village, # length N
  ethnicity = ethnicity,     # length N
  youth = youth#, # length N
)


# compile and run model
mod_pref <- cmdstan_model(tmp_stan_path)

fit_pref <- mod_pref$sample(
  data = stan_data_pref,
  parallel_chains = 4,
  chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000
)



#### Analysis of Market Preferences Model ####

# check model diagnostics
precis(fit_pref)


# reproducibility
set.seed(123)


# plot style
theme_set(theme_minimal())
scale_fill_default <- scale_fill_viridis_d(option = "D", end = 0.85)
scale_color_default <- scale_color_viridis_d(option = "C", end = 0.85)


# posterior draws

  # Raw draws object
draws <- fit_pref$draws(inc_warmup = FALSE)
draws_pref <- fit_pref$draws(inc_warmup = FALSE)


# ---- Person-level theta
theta_draws <- draws %>%
  spread_draws(theta[person])

theta_pref <- theta_draws

stopifnot(all(c(".draw", "person", "theta") %in% names(theta_draws)))


# ---- mu_theta
mu_theta_draws <- draws %>%
  spread_draws(mu_theta)


# ---- Random effects
re_village_draws <- draws %>% spread_draws(re_village[village])
re_youth_draws   <- draws %>% spread_draws(re_youth[youth])
re_eth_draws     <- draws %>% spread_draws(re_ethnicity[ethnicity])
#re_sex_draws     <- draws %>% spread_draws(re_sex[sex])


# ---- Item parameters
a_draws <- draws %>% spread_draws(a[item])
b_draws <- draws %>% spread_draws(b[item])

# ---- PPC draws
# Extract y_rep as a matrix: draws × (N * K)
y_rep_mat <- fit_pref$draws("y_rep", format = "matrix")
stopifnot(ncol(y_rep_mat) == N * K)


# ---- Person metadata
person_data <- tibble(
  person = seq_len(N),
  village_label   = people$village_f,
  youth_label     = people$youth_f,
  ethnicity_label = people$ethnicity_f,
)



#### Figures ####
  # Shared theme 
theme_set(theme_minimal(base_size = 11))

plot_theme <- theme(
  legend.position    = "right",
  panel.grid.minor   = element_blank(),
  axis.text.y        = element_text(size = 12),
  strip.text         = element_text(size = 11)
)

#### > Figure 2a: Posterior densities of random effects (contrasts) ####
youth_effect <- re_youth_draws %>%
  pivot_wider(names_from = youth, values_from = re_youth) %>%
  mutate(effect = `2` - `1`, effect_type = "Adolescent − Adult")

eth_effect <- re_eth_draws %>%
  pivot_wider(names_from = ethnicity, values_from = re_ethnicity) %>%
  mutate(effect = `2` - `1`, effect_type = "Bantu − BaYaka")

village_effect <- re_village_draws %>%
  pivot_wider(names_from = village, values_from = re_village) %>%
  mutate(M_vs_B = `3` - `1`, B_vs_D = `1` - `2`) %>%
  pivot_longer(cols = c(M_vs_B, B_vs_D), names_to = "contrast", values_to = "effect") %>%
  mutate(effect_type = recode(contrast,
                              M_vs_B = "Mankanza − Bokulu",
                              B_vs_D = "Bokulu − Dibumba"))

plot_data <- bind_rows(
  youth_effect   %>% select(effect, effect_type),
  eth_effect     %>% select(effect, effect_type),
  village_effect %>% select(effect, effect_type)
) %>%
  mutate(effect_type = factor(effect_type, levels = c(
    "Adolescent − Adult", "Bantu − BaYaka",
    "Mankanza − Bokulu", "Bokulu − Dibumba"
  )))

x_lim <- max(abs(plot_data$effect)) * c(-0.2, 1)

post_group_contrasts <- ggplot(plot_data, aes(x = effect)) +
  geom_density(fill = "#2C3E50", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ effect_type, ncol = 1, scales = "fixed") +
  labs(x = "Effect on latent market preferences (θ scale)", 
       y = "Density",
       tag = "a") +
  plot_theme +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
post_group_contrasts


#### > Figure 2b: θ posterior by village × ethnicity x adolescent status ####
theta_group_posterior <- theta_draws %>%
  left_join(person_data, by = "person") %>%
  group_by(.draw, village_label, youth_label, ethnicity_label) %>%
  summarise(mean_theta = mean(theta), .groups = "drop") %>%
  mutate(village_label = factor(village_label,
                                levels = c("Dibumba", "Bokulu", "Mankanza")))

theta_post_by_village_group <- ggplot(
  theta_group_posterior,
  aes(x = village_label, y = mean_theta, color = youth_label)
) +
  stat_pointinterval(position = position_dodge(width = 0.4), .width = c(0.50, 0.95)) +
  facet_wrap(~ ethnicity_label) +
  labs(x = NULL, 
       y = expression("Mean latent market orientation (" * theta * ")"),
       tag = "b",
       color = "") +
  plot_theme +
  theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
theta_post_by_village_group


#### > Figure SM1: P(y=1) by item ####
item_labels <- c(
  "lotoko/Apollon", "molenge/beer", "duiker/goat", "wild pig/pig",
  "fish/sardine", "trad work/logging", "honey/cookies", "fruit/candy",
  "drink molenge/beer", "nganga/pharma", "trad learn/school", "trad dance/disco",
  "pusa/peanuts", "palm oil/lotion", "manioc/bread"
)
stopifnot(length(item_labels) == nrow(fit_pref$summary("a")))

summarise_draws <- function(df, value_col) {
  df %>%
    group_by(item) %>%
    summarise(
      med  = median({{ value_col }}),
      lo95 = quantile({{ value_col }}, 0.025),
      hi95 = quantile({{ value_col }}, 0.975),
      lo50 = quantile({{ value_col }}, 0.25),
      hi50 = quantile({{ value_col }}, 0.75),
      .groups = "drop"
    ) %>%
    mutate(item_label = item_labels[item])
}

forest_geoms <- list(
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 0.6, alpha = 0.35),
  geom_linerange(aes(xmin = lo50, xmax = hi50), linewidth = 1.6, alpha = 0.75),
  geom_point(size = 3)
)

post_item_df <- as_draws_df(fit_pref$draws("item_mean_rep")) %>%
  pivot_longer(cols = starts_with("item_mean_rep"), names_to = "item", values_to = "p") %>%
  mutate(item = as.integer(str_extract(item, "\\d+")))

summary_df <- summarise_draws(post_item_df, p) %>%
  mutate(item_label = factor(item_label, levels = rev(item_labels)))


  # plot
item_market_probs <- ggplot(summary_df, aes(y = reorder(item_label, med), x = med)) +
  forest_geoms +
  labs(x = "Posterior P(y=1) (market choice)", 
       y = NULL) +
  plot_theme +
  theme(
    axis.text.y = element_text(size = 16),
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 16)
  )
item_market_probs


#### > Figure SM2: Discrimination (a) by item ####
a_summary <- summarise_draws(a_draws, a)

item_prefs_a <- ggplot(a_summary,
                       aes(x = med, y = reorder(item_label, med))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray60") +
  forest_geoms +
  labs(x = "Discrimination (a)", 
       y = NULL) +
  plot_theme +
  theme(axis.text.y = element_text(size = 16),
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16)
  )
item_prefs_a


# SHARING BREADTH MODEL ####

#### Stan code of Market Preferences Model as string ####
share_stan_code <- "
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
"

# create temporary file path to run Stan model
tmp_stan_path_share <- tempfile(fileext = ".stan")
writeLines(share_stan_code, tmp_stan_path_share)


  #### Run Sharing Breadth Stan model ####
# load data
s <- read_csv("https://raw.githubusercontent.com/ahboyette/nsf_youth_mi_public/main/sm_share_data.csv")


# remove all rows with missing values for item before pivoting
s <- s %>%
  filter(!is.na(item)) %>%      # remove NA items
  distinct(id, item, .keep_all = TRUE)   # remove duplicates


# convert to wide N x K matrix
y <- s %>%
  select(id, item, y) %>%
  pivot_wider(
    names_from = item,
    values_from = y) %>%
  arrange(id) %>%
  select(-id) %>%          # remove id column for Stan
  select(order(as.numeric(names(.)))) %>% # have the columns in order for item_labels
  as.matrix()


# fill in missing values
y[is.na(y)] <- -1 


# Verify 18 columns in matrix
colnames(y) 


N <- nrow(y)  # number of persons
K <- ncol(y)  # number of items
M <- 3        # number of outcome categories


# Prepare per-person demographics
people <- s %>%
  distinct(village, id, youth, ethnicity) %>%  # one row per person
  arrange(id) %>% 
  mutate(
    ethnicity_f = factor(ethnicity, levels = c(1,2), labels = c("BaYaka", "Bantu")),
    youth_f = factor(youth, levels = c(1,2), labels = c("Adult", "Adolescent")),
    village_f = factor(village)
  )


# Convert factors to integers for Stan
ethnicity <- as.integer(people$ethnicity_f)     # 1 = BaYaka, 2 = Bantu
youth <- as.integer(people$youth_f)             # 1 = Adult, 2 = Youth
village <- as.integer(people$village_f)


# Determine group counts
villages <- max(village)
eths     <- max(ethnicity)
ages      <- max(youth)


# prepare Stan list
stan_data_share <- list(
  y = y,
  N = N,
  K = K,
  M = M,
  villages = villages,
  eths = eths,
  ages = ages,
  village = village,
  ethnicity = ethnicity,
  youth = youth
)


# checks
stopifnot(nrow(y) == length(village))
stopifnot(all(y %in% c(-1, 1:M)))
stopifnot(!any(is.na(village)))

table(y[y != -1])

people %>% count(id) %>% filter(n > 1)  # should return empty


# compile and run model
mod_share <- cmdstan_model(tmp_stan_path_share)

fit_share <- mod_share$sample(
  data = stan_data_share,
  parallel_chains = 4,
  chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  adapt_delta = 0.99
)


  #### Analysis of Sharing Breadth model ####

# model diagnostics
  # NOTE that this model will show poor Rhat and bulk effective sample sizes on some runs, but this is stochastic.
precis(fit_share)

# reproducibility
set.seed(123)

# Posterior draws

  # Raw draws object
draws <- fit_share$draws(inc_warmup = FALSE)
draws_shar <- fit_share$draws(inc_warmup = FALSE)


# ---- Person-level theta
theta_draws <- draws %>%
  spread_draws(theta[person])

theta_shar <- theta_draws

stopifnot(all(c(".draw", "person", "theta") %in% names(theta_draws)))


# ---- mu_theta
mu_theta_draws <- draws %>%
  spread_draws(mu_theta)


# ---- Random effects
re_village_draws <- draws %>% spread_draws(re_village[village])
re_youth_draws   <- draws %>% spread_draws(re_youth[youth])
re_eth_draws     <- draws %>% spread_draws(re_ethnicity[ethnicity])


# ---- Item parameters
a_draws <- draws %>% spread_draws(a[item])
b_draws <- draws %>% spread_draws(b[item, threshold])
b_gap_draws     <- draws %>% spread_draws(b_gap[item])
b1_centered_draws <- draws %>% spread_draws(b1_centered[item])


# ---- Add item labels
item_labels <- d %>% select(item, item_name=iname) %>% distinct(item, item_name)
a_draws <- a_draws %>% left_join(item_labels, by = "item")
b_draws <- b_draws %>% left_join(item_labels, by = "item")
b_gap_draws <- b_gap_draws %>% left_join(item_labels, by = "item")
b1_centered_draws <- b1_centered_draws %>% left_join(item_labels, by = "item")


# ---- PPC draws
# Extract y_rep as a matrix: draws × (N * K)
y_rep_mat <- fit_share$draws("y_rep", format = "matrix")
stopifnot(ncol(y_rep_mat) == N * K)


# ---- Person metadata
person_data <- tibble(
  person = seq_len(N),
  village_label   = people$village_f,
  youth_label     = people$youth_f,
  ethnicity_label = people$ethnicity_f
)



  #### Figures ####

# Plot style 
theme_set(theme_minimal(base_size = 11))
plot_theme <- theme(
  legend.position    = "right",
  panel.grid.minor   = element_blank(),
  axis.text.y        = element_text(size = 12),
  strip.text         = element_text(size = 11)
)

#### > Figure 3a. Posterior densities of random effects (contrasts) ####

# Label effects consistently
youth_effect <- re_youth_draws %>%
  pivot_wider(names_from = youth, values_from = re_youth) %>%
  mutate(effect = `2` - `1`,
         effect_type = "Adolescent − Adult")

eth_effect <- re_eth_draws %>%
  pivot_wider(names_from = ethnicity, values_from = re_ethnicity) %>%
  mutate(effect = `2` - `1`,
         effect_type = "Bantu − BaYaka")

village_effect <- re_village_draws %>%
  pivot_wider(names_from = village, values_from = re_village) %>%
  mutate(M_vs_B = `3` - `1`, B_vs_D = `1` - `2`) %>%
  pivot_longer(cols = c(M_vs_B, B_vs_D), names_to = "contrast", values_to = "effect") %>%
  mutate(effect_type = recode(contrast,
                              M_vs_B = "Mankanza − Bokulu",
                              B_vs_D = "Bokulu − Dibumba"))

# combine effects
plot_data <- bind_rows(
  youth_effect   %>% select(effect, effect_type),
  eth_effect     %>% select(effect, effect_type),
  village_effect %>% select(effect, effect_type)
) %>%
  mutate(effect_type = factor(effect_type, levels = c(
    "Adolescent − Adult", 
    "Bantu − BaYaka",
    "Mankanza − Bokulu", 
    "Bokulu − Dibumba"
  )))


# ---- Set symmetric x-axis limits for all facets
x_lim <- max(abs(plot_data$effect)) * c(-0.2, 1)

# ---- Plot with clean, shared x-axis
post_group_contrasts <- ggplot(plot_data, aes(x = effect)) +
  geom_density(fill = "#2C3E50", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ effect_type, ncol = 1, scales = "fixed") +
  labs(
    x = "Effect on narrowing sharing breadth (θ scale)",
    y = "Density",
    tag = "a"
  ) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
post_group_contrasts


#### > Figure 3b. θ posterior by village × ethnicity x adolescent status ####
theta_group_posterior <- theta_draws %>%
  left_join(person_data, by = "person") %>%
  group_by(.draw, village_label, youth_label, ethnicity_label) %>%
  summarise(mean_theta = mean(theta), .groups = "drop") %>%
  mutate(village_label = factor(village_label,
                                levels = c("Dibumba", "Bokulu", "Mankanza")))

theta_post_by_village_group <- ggplot(
  theta_group_posterior,
  aes(x = village_label, y = mean_theta, color = youth_label)
) +
  stat_pointinterval(position = position_dodge(width = 0.4), .width = c(0.50, 0.95)) +
  facet_wrap(~ ethnicity_label) +
  labs(x = NULL, 
       y = expression(atop("Mean latent sharing orientation (" * theta * ")",
                           "(higher means narrower breadth)")),
       tag = "b",
       color = "") +
  plot_theme +
  theme(legend.position = "top", axis.text.x = element_text(angle = 45, hjust = 1))
theta_post_by_village_group


# Item-wise plots for Supplement
# 
# Shared utilities 
forest_geoms <- list(
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 0.6, alpha = 0.35),
  geom_linerange(aes(xmin = lo50, xmax = hi50), linewidth = 1.6, alpha = 0.75),
  geom_point(size = 3)
)

summarise_draws <- function(df, value_col) {
  df %>%
    group_by(item) %>%
    summarise(
      med  = median({{ value_col }}),
      lo95 = quantile({{ value_col }}, 0.025),
      hi95 = quantile({{ value_col }}, 0.975),
      lo50 = quantile({{ value_col }}, 0.25),
      hi50 = quantile({{ value_col }}, 0.75),
      .groups = "drop"
    ) %>%
    left_join(item_labels, by = "item")
}

#### > Figure SM3. Posterior probabilities of each response ####
item_prop_rep_draws <- fit_share$draws("item_prop_rep") %>%
  as_draws_df() %>%
  pivot_longer(
    cols = starts_with("item_prop_rep"),
    names_to = "param",
    values_to = "prop"
  ) %>%
  mutate(
    item     = as.integer(str_extract(param, "(?<=\\[)\\d+")),
    category = as.integer(str_extract(param, "(?<=,)\\d+(?=\\])"))
  )

prop_summary <- item_prop_rep_draws %>%
  group_by(item, category) %>%
  summarise(
    med  = median(prop),
    lo95 = quantile(prop, 0.025),
    hi95 = quantile(prop, 0.975),
    lo50 = quantile(prop, 0.25),
    hi50 = quantile(prop, 0.75),
    .groups = "drop"
  ) %>%
  left_join(item_labels, by = "item") %>%
  mutate(category = factor(category))

item_order <- prop_summary %>%
  filter(category == "3") %>%
  arrange(med) %>%
  pull(item_name)

prop_summary <- prop_summary %>%
  mutate(item_name = factor(item_name, levels = item_order))

# plot
item_share_probs <- ggplot(prop_summary,
                           aes(x = med, y = item_name, color = category)) +
  forest_geoms +
  facet_wrap(~ category, ncol = 3,
             labeller = labeller(category = c(
               "1" = "P(Y=1): Share beyond household",
               "2" = "P(Y=2): Share with household",
               "3" = "P(Y=3): Keep for oneself"
             ))) +
  geom_vline(xintercept = 0.5, linetype = "dotted",
             color = "red", linewidth = 0.5) +
  scale_color_viridis_d() +
  labs(x = "Posterior Probability", 
       y = NULL,
       color = NULL) +
  plot_theme +
  theme(legend.position = "none",
        panel.border = element_rect(color = "grey50", fill = NA,
                                    linewidth = 0.5,
                                    text       = element_text(size = 18),
                                    axis.text.y = element_text(size = 16),
                                    axis.text  = element_text(size = 16),
                                    strip.text = element_text(size = 18)))
item_share_probs


#### > Figure SM4. Discrimination (a) by item ####
a_summary <- a_draws %>%
  group_by(item, item_name) %>%
  summarise(
    med  = median(a),
    lo95 = quantile(a, 0.025),
    hi95 = quantile(a, 0.975),
    lo50 = quantile(a, 0.25),
    hi50 = quantile(a, 0.75),
    .groups = "drop"
  )

item_share_a <- ggplot(a_summary,
                       aes(x = med, y = reorder(item_name, med))) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray60") +
  forest_geoms +
  labs(x = "Discrimination (a)", 
       y = NULL) +
  plot_theme +
  theme(text      = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        axis.text = element_text(size = 14),
        strip.text = element_text(size = 14))
item_share_a