library(ontologyIndex)
library(jsonlite)
library(ontologySimilarity)
library(RCurl)

data(go)
data(gene_GO_terms)
data(GO_IC)

gene_json_data <- jsonlite::fromJSON('Data/landmark_genes.json')

gene_list <- gene_json_data["gene_symbol"][[1]]

beach <- gene_GO_terms[gene_list]

#Go term selection top-5 based on information content (IC)
go_terms <- c()
for (i in (1:length(gene_list) )){
  ic<-get_term_info_content(go,beach[[i]])[0:5]
  go_selected <- beach[[i]][order(ic,decreasing = TRUE)]
  #print(c(i,gene_list[[i]], go_selected))
  go_terms <- c(go_terms, go_selected)
}
go_terms_unique <- sort(unique(go_terms))

#Descriptors binary occurency vector

gene_go_fingerprint <- matrix(0,length(gene_list),length(go_terms_unique))
for (i in (1:length(go_terms_unique))){
  for (j in (1:length(gene_list))){
    if (length(which(beach[[j]] == go_terms_unique[[i]])) == 1){
      gene_go_fingerprint[j,i] <- 1
      
    }
  }
}

rownames(gene_go_fingerprint) <- gene_list
colnames(gene_go_fingerprint) <- go_terms_unique
write.csv(gene_go_fingerprint,file="Data/go_fingerprints.csv", quote=FALSE)
