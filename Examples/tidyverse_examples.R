# Titanic Dataset Analysis — Tidyverse Showcase
# Mirrors the pandas_test.py structure section-for-section.
# Covers: loading, inspection, cleaning, selection, aggregation,
#         grouping, pivoting, joins, strings, windows, viz, export.
#
# Install once: install.packages("tidyverse")  +  install.packages("titanic")

library(tidyverse)   # dplyr, tidyr, stringr, ggplot2, readr, purrr, forcats
library(titanic)     # titanic_train dataset — equivalent to seaborn's load


# ── 1. LOAD ──────────────────────────────────────────────────────────────────

df <- as_tibble(titanic_train)

# Column names to lowercase snake_case to match seaborn version
df <- df %>%
  rename_with(tolower) %>%
  rename(
    passenger_class = pclass,
    siblings_spouses = sibsp,
    ticket_no        = ticket,
  )

cat("Dimensions:", nrow(df), "x", ncol(df), "\n")


# ── 2. INSPECT ───────────────────────────────────────────────────────────────

glimpse(df)                        # dtypes + first values (like df.dtypes + head)

print(head(df))                    # first 6 rows

cat("\n--- descriptive stats (numeric) ---\n")
df %>% select(where(is.numeric)) %>% summary() %>% print()

cat("\n--- missing values per column ---\n")
df %>% summarise(across(everything(), ~ sum(is.na(.)))) %>% print()


# ── 3. SELECTION ─────────────────────────────────────────────────────────────

# Select specific columns
subset_df <- df %>% select(survived, passenger_class, age, sex, fare)

# filter() — boolean / conditional
survived_df     <- df %>% filter(survived == 1)
first_class_w   <- df %>% filter(passenger_class == 1, sex == "female")

# slice() — integer-position selection (like iloc)
row_by_pos <- df %>% slice(1)

# select helpers
numeric_cols <- df %>% select(where(is.numeric))
name_cols    <- df %>% select(starts_with("name"))

cat("\nFirst-class female survivors:",
    nrow(filter(first_class_w, survived == 1)), "\n")


# ── 4. CLEANING ──────────────────────────────────────────────────────────────

df_clean <- df %>%
  # Fill missing age with median
  mutate(age = replace_na(age, median(age, na.rm = TRUE))) %>%
  # Drop rows where embarked is NA
  drop_na(embarked) %>%
  # Remove the cabin column (mostly NA — equivalent to deck in seaborn)
  select(-cabin) %>%
  # Cast survived to logical
  mutate(survived = as.logical(survived)) %>%
  # Deduplicate
  distinct()

cat("\nClean dimensions:", nrow(df_clean), "x", ncol(df_clean), "\n")


# ── 5. FEATURE ENGINEERING ───────────────────────────────────────────────────

df_clean <- df_clean %>%
  mutate(
    family_size = siblings_spouses + parch + 1,
    is_alone    = family_size == 1,
    age_band    = cut(age,
                      breaks = c(0, 12, 18, 35, 60, 100),
                      labels = c("child", "teen", "young adult", "adult", "senior"),
                      right  = TRUE),
    fare_sqrt   = sqrt(fare),          # equivalent to apply(lambda x: x**0.5)
  )


# ── 6. AGGREGATION ───────────────────────────────────────────────────────────

cat("\n--- survival rate by sex ---\n")
df_clean %>%
  group_by(sex) %>%
  summarise(survival_rate = mean(survived)) %>%
  print()

cat("\n--- mean fare & age by class ---\n")
df_clean %>%
  group_by(passenger_class) %>%
  summarise(across(c(fare, age), mean, .names = "mean_{.col}")) %>%
  print()

cat("\n--- fare stats by class ---\n")
df_clean %>%
  group_by(passenger_class) %>%
  summarise(
    min    = min(fare),
    mean   = mean(fare),
    median = median(fare),
    max    = max(fare),
  ) %>%
  print()

cat("\n--- class summary (named aggregations) ---\n")
summary_tbl <- df_clean %>%
  group_by(passenger_class) %>%
  summarise(
    total_passengers = n(),
    survivors        = sum(survived),
    avg_age          = round(mean(age), 2),
    avg_fare         = round(mean(fare), 2),
  )
print(summary_tbl)

# transform equivalent — broadcast group stat back to each row
df_clean <- df_clean %>%
  group_by(passenger_class) %>%
  mutate(class_mean_fare = mean(fare)) %>%
  ungroup()

cat("\n--- embarkation counts ---\n")
df_clean %>% count(embarked, sort = TRUE) %>% print()


# ── 7. PIVOT / RESHAPE ───────────────────────────────────────────────────────

cat("\n--- survival rate pivot (class × sex) ---\n")
pivot_tbl <- df_clean %>%
  group_by(passenger_class, sex) %>%
  summarise(survival_rate = round(mean(survived), 3), .groups = "drop") %>%
  pivot_wider(names_from = sex, values_from = survival_rate)
print(pivot_tbl)

# pivot_longer() — wide → long  (equivalent to melt())
cat("\n--- melted (long format) ---\n")
pivot_tbl %>%
  pivot_longer(cols = c(female, male), names_to = "sex", values_to = "survival_rate") %>%
  head() %>%
  print()

# tabyl-style crosstab using dplyr + pivot_wider
cat("\n--- crosstab: class vs survived (row %) ---\n")
df_clean %>%
  count(passenger_class, survived) %>%
  group_by(passenger_class) %>%
  mutate(pct = round(n / sum(n), 3)) %>%
  select(-n) %>%
  pivot_wider(names_from = survived, values_from = pct) %>%
  print()


# ── 8. SORTING & RANKING ─────────────────────────────────────────────────────

cat("\n--- top 5 fares ---\n")
df_clean %>%
  slice_max(fare, n = 5) %>%
  select(name, passenger_class, fare) %>%
  print()

df_clean <- df_clean %>%
  mutate(fare_rank = min_rank(desc(fare)))


# ── 9. JOIN ──────────────────────────────────────────────────────────────────

port_names <- tibble(
  embarked = c("S", "C", "Q"),
  port     = c("Southampton", "Cherbourg", "Queenstown"),
)

df_merged <- df_clean %>% left_join(port_names, by = "embarked")
cat("\nPort column added:", "port" %in% names(df_merged), "\n")


# ── 10. STRING OPERATIONS ────────────────────────────────────────────────────

df_merged <- df_merged %>%
  mutate(
    sex_upper      = str_to_upper(sex),
    name_title     = str_extract(name, "(Mr|Mrs|Miss|Master)\\."),
    name_length    = str_length(name),
  )


# ── 11. WINDOW FUNCTIONS ─────────────────────────────────────────────────────

df_windows <- df_merged %>%
  arrange(fare) %>%
  mutate(
    fare_rolling_mean = slide_dbl(fare, mean, .before = 9, .complete = FALSE),  # see note
    fare_cumsum       = cumsum(fare),
  )
# Note: slide_dbl requires the 'slider' package. If unavailable, use:
# fare_rolling_mean = zoo::rollmean(fare, k = 10, fill = NA, align = "right")


# ── 12. VISUALISATION (ggplot2) ──────────────────────────────────────────────

# Survival rate by class & sex
p1 <- pivot_tbl %>%
  pivot_longer(c(female, male), names_to = "sex", values_to = "survival_rate") %>%
  ggplot(aes(x = factor(passenger_class), y = survival_rate, fill = sex)) +
  geom_col(position = "dodge") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Survival rate by class & sex", x = "Class", y = "Rate") +
  theme_minimal()

# Age distribution
p2 <- df_clean %>%
  ggplot(aes(x = age)) +
  geom_histogram(bins = 30, fill = "steelblue", colour = "white") +
  labs(title = "Age distribution", x = "Age") +
  theme_minimal()

# Fare by class (box plot)
p3 <- df_clean %>%
  ggplot(aes(x = factor(passenger_class), y = fare)) +
  geom_boxplot(fill = "lightgrey") +
  labs(title = "Fare by class", x = "Class", y = "Fare") +
  theme_minimal()

# Survival rate by family size
p4 <- df_clean %>%
  group_by(family_size) %>%
  summarise(survival_rate = mean(survived)) %>%
  ggplot(aes(x = factor(family_size), y = survival_rate)) +
  geom_col(fill = "coral") +
  labs(title = "Survival rate by family size", x = "Family size", y = "Rate") +
  theme_minimal()

# Combine into a 2x2 grid and save
library(patchwork)   # install.packages("patchwork")
combined <- (p1 | p2) / (p3 | p4) +
  plot_annotation(title = "Titanic — Tidyverse Analysis")

ggsave("titanic_analysis_r.png", combined, width = 12, height = 9, dpi = 120)
cat("\nPlot saved to titanic_analysis_r.png\n")


# ── 13. EXPORT ───────────────────────────────────────────────────────────────

write_csv(summary_tbl, "class_summary_r.csv")
jsonlite::write_json(summary_tbl, "class_summary_r.json", pretty = TRUE)
cat("Exported class_summary_r.csv and class_summary_r.json\n")

cat("\nDone.\n")
