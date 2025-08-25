library(enrichR)
library(stringr)
library(multienrichjam)
library(DOSE)
library(argparse)
library(dplyr)
library(purrr)


#####

rm(list=ls())

parser <- ArgumentParser(description='Define directories')
parser$add_argument('--input_dir', type="character", 
                    default = "/Users/svitlana.oleshko/Projects/biopathnet/revision2/paths",
                    help='Directory with saved summary of top 10 most important paths for top predictions')

args <- parser$parse_args()

input_dir <- paste0(args$input_dir, "/")

#####

get_consistent_genes <- function(setting){

  file_path <- paste0(input_dir, "genes_", setting, ".csv")
  
  if (!file.exists(file_path)) {
    message("File does not exist: ", file_path)
    return(NULL)
  }
  
  genes_df <- data.table::fread(paste0(input_dir, "genes_", setting, ".csv")) 
  n_total_seed <- length(unique(genes_df$seed))
  
  # extract number of seeds this gene occurs in the most important paths
  genes_df <- genes_df %>%
    group_by(gene_id) %>%
    summarise(n_seed = n()) 
  
  # select genes that occur in the most important paths for the most seeds
  genes <- genes_df$gene_id[genes_df$n_seed > n_total_seed * 0.5]
  
  return(genes)
}


# Define lists of genes for which check enrichment in databases
genes_list <- list("genes_GF_general" = get_consistent_genes("genefunc"),
                   "genes_DR_splitAG" = get_consistent_genes("adrenal_gland"),
                   "genes_DR_splitAN" = get_consistent_genes("anemia"),
                   "genes_DR_splitCP" = get_consistent_genes("cell_proliferation"),
                   "genes_DR_splitCV" = get_consistent_genes("cardiovascular"),
                   "genes_DR_splitMH" = get_consistent_genes("mental_health"),
                   "genes_SL_thr030" = get_consistent_genes("SL_thr030"),
                   "genes_SL_thr000" = get_consistent_genes("SL_thr000"),
                   "genes_LT_general" = get_consistent_genes("lnctard")
                   )

genes_list <- Filter(Negate(is.null), genes_list)


# Define databases 
dbs <- c("KEGG_2021_Human", "DisGeNET")

# Perform enrichment analysis

i_dbs <- seq(1, length(dbs))
i_genesets <- seq(1, length(genes_list))

extract_enrichment <- function(gene_list, dbs, i_genesets, i_dbs) {
  enriched <- enrichr(gene_list[[i_genesets]], dbs)
  
  if (i_dbs <= length(enriched)) {
    enriched_df <- enriched[[i_dbs]]
    return(enriched_df)
  } else {
    print("Error")
  }
}

result_lst <- list()
for (i in 1:length(genes_list)){
  for (j in (1:length(dbs))){
    enriched <- enrichr(genes_list[[i]], dbs)
    res_df <- enriched[[j]]
    name <- paste0(names(genes_list)[i], "_", dbs[j])
    result_lst[[name]] <- res_df
  }
}

 
combined_df <- do.call(rbind, lapply(names(result_lst), function(name) {
  df <- result_lst[[name]]
  df <- df %>% mutate(source = name,
                      task = str_extract(source, "GF|DR|SL|LT"),
                      setting = str_extract(source, "thr\\d+|split[A-Z]+|general+"),
                      database = str_remove(source, "genes_(GF|DR|SL|LT)_(general+|thr\\d+|split[A-Z]+)_"),
                      task = case_when(
                        task == "GF" & setting == "general" ~ "Gene function prediction",
                        task == "SL" & setting == "thr030" ~ "SL prediction (thr = 0.3)",
                        task == "SL" & setting == "thr000" ~ "SL prediction (thr = 0.0)",
                        task == "DR" & setting == "splitAG" ~ "Drug repurposing (AG)",
                        task == "DR" & setting == "splitAN" ~ "Drug repurposing (AN)",
                        task == "DR" & setting == "splitCP" ~ "Drug repurposing (CP)",
                        task == "DR" & setting == "splitCV" ~ "Drug repurposing (CV)",
                        task == "DR" & setting == "splitMH" ~ "Drug repurposing (MH)",
                        task == "LT" & setting == "general" ~ "LcnRNA-target prediction",
                        TRUE ~ task 
                      )
                      ) %>%
    dplyr::select(-c(source, setting))
  return(df)
}))


# Plot enrichment analysis results

ora_dotplot_df <- function(plt_df){
  plt_df$geneHits <- sapply(plt_df$Overlap, function(x) {
    parts <- as.numeric(unlist(strsplit(x, "/")))
    parts[1]
  })
  
  plt_df$geneHitsRatio <- sapply(plt_df$Overlap, function(x) {
    parts <- as.numeric(unlist(strsplit(x, "/")))
    parts[1] / parts[2]
  })
  
  top_10_df <- plt_df[1:10,] %>%
    mutate(sign = "Top 10 terms")
  
  bottom_10_df <- plt_df[(nrow(plt_df) - 9):nrow(plt_df),] %>%
    mutate(sign = "Bottom 10 terms")
  
  topbottom_10_df <- rbind(top_10_df, bottom_10_df)
  
  return(topbottom_10_df)
}


SL_thr030_df <- ora_dotplot_df(combined_df %>% filter(task == "SL prediction (thr = 0.3)" & database == "KEGG_2021_Human"))
DR_AG_df <- ora_dotplot_df(combined_df %>% filter(task == "Drug repurposing (AG)" & database == "DisGeNET"))
DR_AN_df <- ora_dotplot_df(combined_df %>% filter(task == "Drug repurposing (AN)" & database == "DisGeNET"))
DR_CP_df <- ora_dotplot_df(combined_df %>% filter(task == "Drug repurposing (CP)" & database == "KEGG_2021_Human"))
DR_CV_df <- ora_dotplot_df(combined_df %>% filter(task == "Drug repurposing (CV)" & database == "DisGeNET"))
DR_MH_df <- ora_dotplot_df(combined_df %>% filter(task == "Drug repurposing (MH)" & database == "DisGeNET"))

combined_df <- rbind(SL_thr030_df, DR_CP_df, DR_MH_df, DR_AG_df, DR_AN_df, DR_CV_df)


ora_dotplot <- function(plt_df, tsk, breaks){
  
  plt_df <- plt_df %>% filter(task == tsk)
  
  topbottom_10_eres <- enrichDF2enrichResult(enrichDF = plt_df,
                                             pvalueCutoff = 1,
                                             pAdjustMethod = "none",
                                             keyColname = "Term",
                                             geneHits = "geneHits",
                                             geneRatioColname = "Overlap",
                                             pvalueColname = "Adjusted.P.value",
                                             descriptionColname = "Term"
  )
  
  hits_ratios <- sapply(plt_df$Overlap, function(x) {
    parts <- as.numeric(unlist(strsplit(x, "/")))
    parts[1] / parts[2]
  })
  
  plt <- dotplot(topbottom_10_eres, x = "geneRatio", 
                 color = "p.adjust", showCategory=20, 
                 font.size = 12,  title = tsk,
                 label_format = 50) +
    scale_x_continuous(breaks = seq(0, 
                                    (ceiling(max(hits_ratios) * 100))/100, 
                                    by = breaks)) + 
    geom_hline(yintercept = 10.4, linetype = "dashed", color = "black") 
  
  return(plt)
}


task_db_list <- list(
  "SL prediction (thr = 0.3)" = "KEGG_2021_Human",
  "Drug repurposing (AG)" = "DisGeNET",
  "Drug repurposing (AN)" = "DisGeNET",
  "Drug repurposing (CP)" = "KEGG_2021_Human",
  "Drug repurposing (CV)" = "DisGeNET",
  "Drug repurposing (MH)" = "DisGeNET"
)

filtered_dfs <- imap(task_db_list, function(db, tsk) {
  combined_df %>%
    filter(task == tsk, database == db) %>%
    ora_dotplot_df()
})

plot_df <- bind_rows(filtered_dfs)

ora_dotplot(plot_df, "SL prediction (thr = 0.3)", breaks = 0.02)
ora_dotplot(plot_df, "Drug repurposing (MH)", breaks = 0.02)
ora_dotplot(plot_df, "Drug repurposing (CP)", breaks = 0.05)
ora_dotplot(plot_df, "Drug repurposing (AG)", breaks = 0.05)
ora_dotplot(plot_df, "Drug repurposing (AN)", breaks = 0.1)
ora_dotplot(plot_df, "Drug repurposing (CV)", breaks = 0.01)

