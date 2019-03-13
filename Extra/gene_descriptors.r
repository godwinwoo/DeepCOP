library(jsonlite)
library(ontologySimilarity)

data(gene_GO_terms) # Gene Ontology annotation of genes
gene_json_data <- jsonlite::fromJSON('landmark_genes.json')
gene_symbols <- gene_json_data["gene_symbol"][[1]]
landmark_genes_GO_terms <- gene_GO_terms[gene_symbols]

# gather all go terms for the lm genes
all_landmark_genes_go_terms_vector <- c()
for (i in (1:length(gene_symbols) )){
  go_selected <- landmark_genes_GO_terms[[i]]
  all_landmark_genes_go_terms_vector <- c(all_landmark_genes_go_terms_vector, go_selected)
}
all_unique_landmark_genes_go_terms <- sort(unique(all_landmark_genes_go_terms_vector))

# get go terms that are attributed to by more than 3 landmark genes
common_go_terms <- c()
for (i in (1:length(all_unique_landmark_genes_go_terms))){
  lm_go_term<-all_unique_landmark_genes_go_terms[[i]]
  count <- 0
  for (j in (1:length(gene_symbols))){
    lm_gene_go_terms <- landmark_genes_GO_terms[[j]]
    if (any(lm_gene_go_terms==lm_go_term)){
      count <- count+1
    }
    if (count > 3){
      common_go_terms <- c(common_go_terms, lm_go_term)
      break
    }
  }
}

# descriptors binary occurency vector
gene_go_fingerprint <- matrix(0,length(gene_symbols),length(common_go_terms))
for (i in (1:length(common_go_terms))){
  for (j in (1:length(gene_symbols))){
    if (length(which(landmark_genes_GO_terms[[j]] == common_go_terms[[i]])) == 1){
      gene_go_fingerprint[j,i] <- 1
    }
  }
}

rownames(gene_go_fingerprint) <- gene_symbols
colnames(gene_go_fingerprint) <- common_go_terms
write.csv(gene_go_fingerprint,file="go_fingerprints.csv", quote=FALSE)
