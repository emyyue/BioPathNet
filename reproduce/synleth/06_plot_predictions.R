# Load necessary libraries
library(stringr)
library(dplyr)
library(tidyverse)
library(data.table)
library(patchwork)
library(ggplot2)
library(purrr)
library(ggthemes)
library(grid)
library(rstatix)
library(ggpubr)
library(sigmoid)
library(gridExtra)
library(kableExtra)
library(DT)

# Load data

base_path <- "./reproduce/synleth/"

preds_1234 <- fread(paste0(base_path, "output/Predictions_BioPathNet_threshold_030_seed_1234.csv"))
preds_1235 <- fread(paste0(base_path, "output/Predictions_BioPathNet_threshold_030_seed_1235.csv"))
preds_1236 <- fread(paste0(base_path, "output/Predictions_BioPathNet_threshold_030_seed_1236.csv"))
preds_1237 <- fread(paste0(base_path, "output/Predictions_BioPathNet_threshold_030_seed_1237.csv"))
preds_1238 <- fread(paste0(base_path, "output/Predictions_BioPathNet_threshold_030_seed_1238.csv"))

preds <- rbind(preds_1234, preds_1235, preds_1236, preds_1237, preds_1238)

entities <- fread(paste0(base_path, "KR4SL_thr030/data/entities.txt"))
test_data_kr4sl <- fread(paste0(base_path, "KR4SL_thr030/data/test_filtered.txt"), header = FALSE)
train_data_kr4sl <- fread(paste0(base_path, "KR4SL_thr030/data/train_filtered.txt"), header = FALSE)
valid_data_kr4sl <- fread(paste0(base_path, "KR4SL_thr030/data/valid_filtered.txt"), header = FALSE)
fact_data_kr4sl <- fread(paste0(base_path, "KR4SL_thr030/data/facts.txt"), header = FALSE)

# Prepare data

names(entities) <- c('entity_name', 'entity_id', 'entity_type')
entities$entity_id <- as.factor(entities$entity_id)

test_data_kr4sl <- test_data_kr4sl %>% mutate(relation = paste0(V1, "-", V3))
train_data_kr4sl <- train_data_kr4sl %>% mutate(relation = paste0(V1, "-", V3))
valid_data_kr4sl <- valid_data_kr4sl %>% mutate(relation = paste0(V1, "-", V3))
fact_data_kr4sl <- fact_data_kr4sl %>% filter(V2 == "SL_GsG") %>%
  mutate(relation = paste0(V1, "-", V3)) %>%
  bind_rows(
    select(fact_data_kr4sl, V3, V2, V1) %>%
      mutate(relation = paste0(V3, "-", V1)) %>%
      rename(V1 = V1, V2 = V2, V3 = V3)
  ) %>% distinct()

known_train_sl_pairs <- bind_rows(train_data_kr4sl, valid_data_kr4sl, fact_data_kr4sl)
known_test_sl_pairs <- test_data_kr4sl

# Process predictions

process_predictions <- function(pred_df, selected_genes, top_k, seed) {
  pred_df %>%
    filter(gene_symbol %in% selected_genes, seed == seed) %>%
    select(-seed) %>%
    melt(id.vars = "gene_symbol") %>%
    mutate(entity_id = variable) %>%
    left_join(entities, by = "entity_id") %>%
    mutate(
      relation = paste0(gene_symbol, "-", entity_name),
      SL_pair = case_when(
        relation %in% known_test_sl_pairs$relation ~ "known (test)",
        relation %in% known_train_sl_pairs$relation ~ "known (train)",
        TRUE ~ "unknown"
      )
    ) %>%
    split(.$gene_symbol) %>%
    map(~ .x %>%
          arrange(desc(value)) %>%
          slice_head(n = top_k) %>%
          mutate(SL_pair = as.factor(SL_pair), entity_name = as.factor(entity_name))
    )
}


seeds <- c(1234, 1235, 1236, 1237, 1238)
selected_genes <- c("EYA4", "POLB")
top_k <- 20

results <- map(seeds, ~ process_predictions(preds, selected_genes, top_k, .x))

results <- map(results, ~ map(.x, ~ mutate(.x, prob = sigmoid(value))))

# Prepare data for plotting

plot_data <- function(pred_data) {
  bind_rows(pred_data, .id = "seed") %>%
    mutate(SL_pair = ifelse(SL_pair == "unknown", "unknown", "known"))
}

# Create plots for each seed

plots <- map(seeds, ~ {
  plot_data(results[[which(seeds == .x)]]) %>%
    ggplot(aes(x = reorder(gene_symbol, -value), y = prob, fill = SL_pair)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = paste("Predictions for Seed", .x), x = "Gene Symbol", y = "Probability") +
    theme_minimal()
})

wrap_plots(plots)
