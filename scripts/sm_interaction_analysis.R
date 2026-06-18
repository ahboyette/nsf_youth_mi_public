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


# MARKET PREFERENCES INTERACTION MODEL ####

  # H2a between adolescent status and village
  # H4a between ethnicity and village

s <- read_csv("https://raw.githubusercontent.com/ahboyette/nsf_youth_mi_public/main/data/sm_pref_data.csv")




#### Compile and fit the interaction model ####
mod_interaction <- cmdstan_model("~/Nextcloud/Projects/NSF/Publication/schemas/prefs_model_interaction.stan")






fit_interaction_eth_village <- mod_interaction$sample(
  data = stan_data,
  parallel_chains = 4,
  chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  seed = 123
)


##### youth (adolescent status) x village ####
precis(fit_interaction_youth_village)
#                      mean   sd  5.5% 94.5% rhat ess_bulk
# sigma_village        0.68 0.36  0.28  1.38    1  4596.11
# sigma_ethnicity      0.38 0.38  0.02  1.13    1  4052.41
# sigma_youth          0.63 0.44  0.14  1.46    1  4583.92
# sigma_youth_village  0.14 0.13  0.01  0.39    1  2863.64
# sigma_person         0.56 0.08  0.44  0.69    1  1904.55
# mu_theta            -0.62 0.60 -1.61  0.29    1  4950.41
# grand_mean           0.00 0.08 -0.11  0.11    1 10484.91


# Extract interaction draws
draws <- fit_interaction_youth_village$draws()
int_draws <- draws %>%
  spread_draws(re_youth_village[age, village])

village_labels <- levels(people$village_f)

# Compute difference-in-differences
interaction_effect <- draws %>%
  spread_draws(re_youth_village[age, village]) %>%
  filter(is.finite(re_youth_village)) %>%
  group_by(.draw, village) %>%
  summarise(
    youth_effect = re_youth_village[age == 2] - 
      re_youth_village[age == 1],
    .groups = "drop"
  )

ggplot(interaction_effect,
       aes(x = youth_effect, y = factor(village, labels = village_labels))) +
  stat_halfeye(fill = "steelblue") +
  labs(
    x = "Youth effect (Youth − Adult)",
    y = "Village",
    title = "Youth effect by village (interaction)"
  ) +
  theme_minimal()


int_draws <- draws %>%
  spread_draws(re_youth_village[age, village])

int_draws %>%
  ungroup() %>%
  filter(is.finite(re_youth_village)) %>%
  mutate(
    youth = factor(age, levels = c(1, 2), labels = c("Adult", "Youth")),
    village = factor(village)
  ) %>%
  ggplot(aes(x = re_youth_village, y = factor(village, labels = village_labels), fill = youth)) +
  stat_halfeye(position = position_dodge(width = 0.6)) +
  labs(
    x = "Effect on θ",
    y = "Village",
    title = "Youth × Village interaction"
  ) +
  theme_minimal()

# there is a slight shift right for youth and left for adults in Djoube, but the effect is tiny.


ggplot(interaction_effect,
       aes(x = youth_effect, y = factor(village, labels = village_labels))) +
  stat_halfeye(fill = "steelblue") +
  labs(
    x = "Youth effect (Youth − Adult)",
    y = "Village",
    title = "Youth × Village interaction"
  ) +
  theme_minimal()



##### ethnicity x village ####
precis(fit_interaction_eth_village)

#                    mean   sd  5.5% 94.5% rhat ess_bulk
# sigma_village      0.62 0.37  0.20  1.31    1  3317.73
# sigma_ethnicity    0.38 0.39  0.02  1.14    1  3034.11
# sigma_youth        0.65 0.41  0.20  1.42    1  4434.79
# sigma_eth_village  0.17 0.13  0.02  0.41    1  1830.38
# sigma_person       0.54 0.08  0.42  0.68    1  1329.42
# mu_theta          -0.61 0.58 -1.56  0.29    1  3696.72
# grand_mean         0.00 0.09 -0.13  0.14    1  7831.94


# Extract interaction draws
draws <- fit_interaction_eth_village$draws()
int_draws <- draws %>%
  spread_draws(re_eth_village[ethnicity, village])

# Compute difference-in-differences
interaction_effect <- draws %>%
  spread_draws(re_eth_village[ethnicity, village]) %>%
  filter(is.finite(re_eth_village)) %>%
  group_by(.draw, village) %>%
  summarise(
    eth_effect = re_eth_village[ethnicity == 2] - 
      re_eth_village[ethnicity == 1],
    .groups = "drop"
  )

ggplot(interaction_effect,
       aes(x = eth_effect, y = factor(village))) +
  stat_halfeye(fill = "steelblue") +
  labs(
    x = "Ethnicity effect (Bantu − BaYaka)",
    y = "Village",
    title = "Ethnicity effect by village (interaction)"
  ) +
  theme_minimal()


int_draws <- draws %>%
  spread_draws(re_eth_village[ethnicity, village])

int_draws %>%
  ungroup() %>%
  filter(is.finite(re_eth_village)) %>%
  mutate(
    youth = factor(ethnicity, levels = c(1, 2), labels = c("BaYaka", "Bantu")),
    village = factor(village)
  ) %>%
  ggplot(aes(x = re_eth_village, y = village, fill = ethnicity)) +
  stat_halfeye(position = position_dodge(width = 0.6)) +
  labs(
    x = "Effect on θ",
    y = "Village",
    title = "Ethnicity × Village interaction"
  ) +
  theme_minimal()


village_labels <- levels(people$village_f)

ggplot(interaction_effect,
       aes(x = eth_effect, y = factor(village, labels = village_labels))) +
  stat_halfeye(fill = "steelblue") +
  labs(
    x = "Ethnicity effect (Bantu − BaYaka)",
    y = "Village",
    title = "Ethnicity × Village interaction"
  ) +
  theme_minimal()



# compare the models ####
# Compute LOO for both models
loo_additive <- fit$loo()  # your original model
loo_interaction_youth_eth <- fit_interaction_youth_eth$loo()
loo_interaction_youth_village <- fit_interaction_youth_village$loo()
loo_interaction_eth_village <- fit_interaction_eth_village$loo()

# Compare
library(loo)
loo_compare(loo_additive, 
            loo_interaction_youth_eth,
            loo_interaction_youth_village,
            loo_interaction_eth_village)

#        elpd_diff se_diff
# model4  0.0       0.0   
# model3 -0.6       0.8   # tied
# model2 -0.9       0.9   # tied
# model1 -1.4       0.9   # tied

# none is clearly better. The diff is less than 2x the error
