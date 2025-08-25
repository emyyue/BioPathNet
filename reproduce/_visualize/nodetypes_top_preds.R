library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(argparse)


#####

rm(list=ls())

parser <- ArgumentParser(description='Define directories')
parser$add_argument('--input_dir', type="character", 
                    default = "/Users/svitlana.oleshko/Projects/biopathnet/revision2/node_types",
                    help='Directory with saved summary of top 10 most important paths for top predictions')

args <- parser$parse_args()

input_dir <- paste0(args$input_dir, "/")

#####

# Define node type mappings

node_type_map_SL <- c("0" = "BP", 
                      "1" = "CC", 
                      "2" = "gene", 
                      "3" = "MF", 
                      "4" = "pathway")

node_type_map_DR <- c("0" = "anatomy", 
                      "1" = "BP", 
                      "2" = "CC", 
                      "3" = "disease", 
                      "4" = "drug",
                      "5" = "phenotype", 
                      "6" = "exposure", 
                      "7" = "gene", 
                      "8" = "MF", 
                      "9" = "pathway")

# Load and prepare data

load_node_data <- function(filename, task_label, node_map) {
  file_path <- file.path(input_dir, filename)
  fread(file_path) %>%
    mutate(
      node_type = node_map[as.character(node_type)],
      task = task_label
    )
}

nts_allseeds <- bind_rows(
  load_node_data("SL_all_seeds.csv", "SL prediction (thr = 0.3)", node_type_map_SL),
  load_node_data("cell_proliferation_all_seeds.csv", "Drug repurposing (CP)", node_type_map_DR),
  load_node_data("adrenal_gland_all_seeds.csv", "Drug repurposing (AG)", node_type_map_DR),
  load_node_data("anemia_all_seeds.csv", "Drug repurposing (AN)", node_type_map_DR),
  load_node_data("cardiovascular_all_seeds.csv", "Drug repurposing (CV)", node_type_map_DR),
  load_node_data("mental_health_all_seeds.csv", "Drug repurposing (MH)", node_type_map_DR)
) 


df_reduced <- nts_allseeds %>%
  group_by(node_type, task) %>%
  arrange(desc(num_paths_with_node), .by_group = TRUE) %>%
  mutate(
    rank = row_number(),
    node_name = ifelse(rank > 5, "Other", node_name),
    node_rank = factor(ifelse(rank > 5, "> 5", rank), levels = c("1", "2", "3", "4", "5", "> 5")),
    node_name_adj = gsub("_", " ", node_name)
  ) %>%
  group_by(node_type, node_name, node_name_adj, node_rank, task) %>%
  summarise(
    num_paths_with_node = sum(num_paths_with_node),
    num_nodes_of_type = mean(num_nodes_of_type),
    .groups = "drop"
  )


# Prepare data for plotting

first_node_type_per_task <- df_reduced %>%
  group_by(task, node_type) %>%
  summarise(total = sum(num_paths_with_node), .groups = "drop") %>%
  group_by(task) %>%
  slice_max(order_by = total, n = 2, with_ties = FALSE)

df_tagged <- df_reduced %>%
  left_join(first_node_type_per_task, by = c("task", "node_type")) %>%
  mutate(is_top = !is.na(total))


prepare_plot_data <- function(df, task_label, label_scale = 1000, label_angle = 70) {
  df_plot <- df %>% filter(task == task_label)
  
  df_first <- df_plot %>% filter(is_top)
  
  df_rest <- df_plot %>% filter(!is_top) %>%
    mutate(label_position = num_nodes_of_type + (7 - as.numeric(node_rank)) * label_scale)
  
  node_order <- df_plot %>%
    group_by(node_type) %>%
    summarise(total = mean(num_nodes_of_type), .groups = "drop") %>%
    arrange(desc(total)) %>%
    pull(node_type)
  
  df_plot$node_type <- factor(df_plot$node_type, levels = node_order)
  df_first$node_type <- factor(df_first$node_type, levels = node_order)
  df_rest$node_type <- factor(df_rest$node_type, levels = node_order)
  
  top_node_types <- node_order[1:2]
  
  df_first_updated <- lapply(top_node_types, function(nt) {
    assign_y_positions(df_first, nt)
  }) %>% bind_rows()
  
  df_first <- df_first_updated
  
  ordered_ranks <- c("1", "2", "3", "4", "5", "> 5")
  df_plot$node_rank <- factor(df_plot$node_rank, levels = ordered_ranks)
  df_first_updated$node_rank <- factor(df_first_updated$node_rank, levels = ordered_ranks)
  
  
  
  list(df_plot = df_plot, df_first = df_first_updated, df_rest = df_rest, angle = label_angle)
}

assign_y_positions <- function(df_first, node_type) {
  df_subset <- df_first %>%
    filter(node_type == !!node_type) %>%
    arrange(num_paths_with_node)
  
  other <- df_subset %>% filter(node_name_adj == "Other")
  top_nodes <- df_subset %>% filter(node_name_adj != "Other")
  
  y_pos_other <- if (nrow(other) == 1) other$num_paths_with_node / 2 else 0
  y_pos_top <- cumsum(top_nodes$num_paths_with_node) - top_nodes$num_paths_with_node / 2
  
  top_nodes <- top_nodes %>%
    mutate(y_position = y_pos_top + 2*y_pos_other)
  
  other <- other %>%
    mutate(y_position = y_pos_other)
  
  bind_rows(top_nodes, other)
}


plot_node_types <- function(df_plot, df_first, df_rest, angle) {
  ggplot(df_plot, aes(x = node_type, y = num_paths_with_node)) +
    geom_bar(aes(fill = node_type, alpha = node_rank), 
             stat = "identity", color = "black", linewidth = 0.1) +
    geom_text(data = df_first,
              aes(x = node_type, y = y_position, label = node_name_adj),
              # position = position_stack(vjust = 0.5),
              inherit.aes = FALSE,
              color = "black", size = 1.5) +
    geom_text(data = df_rest,
              aes(x = node_type, y = label_position, label = node_name_adj),
              vjust = 0, hjust = 0.4, angle = angle, size = 1.5) +
    scale_alpha_manual(values = c("1" = 0.9, "2" = 0.8, "3" = 0.7, 
                                  "4" = 0.6, "5" = 0.5, "> 5" = 1.0)) +
    scale_fill_manual(values = c(
      "anatomy" = "#8f5682", 
      "BP" = "#f9844a", 
      "CC" = "#577590", 
      "disease" = "#277da1",
      "drug" = "#f3722c", 
      "phenotype" = "#f94144", 
      "exposure" = "#43aa8b", 
      "gene" = "#f9c74f",
      "MF" = "#f8961e", 
      "pathway" = "#DB2E34")) +
    labs(x = "Node type", y = "Node Occurrences in Top Explanation Paths") +
    theme_bw() +
    theme(
      legend.position = "right",
      text = element_text(size = 10),
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
    ) +
    guides(alpha = "none", 
           # color = guide_legend(title = "Node type"), 
           fill = guide_legend(title = "Node type")) +
    facet_wrap(~task, scales = "free")
}


plot_data <- prepare_plot_data(df_tagged, "SL prediction (thr = 0.3)", label_scale = 1000, label_angle = 70)
p1 <- plot_node_types(plot_data$df_plot, plot_data$df_first, plot_data$df_rest, plot_data$angle)


plot_data <- prepare_plot_data(df_tagged, "Drug repurposing (AG)", label_scale = 700, label_angle = 70)
p2 <- plot_node_types(plot_data$df_plot, plot_data$df_first, plot_data$df_rest, plot_data$angle)

plot_data <- prepare_plot_data(df_tagged, "Drug repurposing (AN)", label_scale = 300, label_angle = 70)
p3 <- plot_node_types(plot_data$df_plot, plot_data$df_first, plot_data$df_rest, plot_data$angle)

plot_data <- prepare_plot_data(df_tagged, "Drug repurposing (CP)", label_scale = 700, label_angle = 70)
p4 <- plot_node_types(plot_data$df_plot, plot_data$df_first, plot_data$df_rest, plot_data$angle)

plot_data <- prepare_plot_data(df_tagged, "Drug repurposing (CV)", label_scale = 800, label_angle = 75)
p5 <- plot_node_types(plot_data$df_plot, plot_data$df_first, plot_data$df_rest, plot_data$angle)

plot_data <- prepare_plot_data(df_tagged, "Drug repurposing (MH)", label_scale = 300, label_angle = 70)
p6 <- plot_node_types(plot_data$df_plot, plot_data$df_first, plot_data$df_rest, plot_data$angle)
