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
stan_data <- list(
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


# run model
mod <- cmdstan_model(
  file = "https://raw.githubusercontent.com/ahboyette/nsf_youth_mi_public/main/sm_pref_model.stan"
) 

fit_pref <- mod$sample(
  data = stan_data,
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
  labs(x = "Effect on latent market orientation (θ scale)", 
       y = "Density",
       tag = "a") +
  plot_theme +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
post_group_contrasts


#### > Figure 2b: θ posterior by group × village × ethnicity ####
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


#### Figure SM2: Discrimination (a) by item ####
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


