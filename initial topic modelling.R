

library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(textclean)
library(textstem)
library(topicmodels)
library(ldatuning)
library(tm)
library(ggplot2)
library(emdbook)
library(doParallel)



###############################
######## DATA CLEANING ########
###############################

## tokenizing data, removing stopwords, and lemmatizing words

essays %>%
  unnest_tokens(input    = Essay_Text,
                output   = word,
                token    = "words",
                drop     = FALSE,
                to_lower = FALSE) %>%
  mutate(word = str_replace(word, "\\'s$", ""),
         word = lemmatize_words(word),
         word = replace_number(word)) %>%
  anti_join(y  = stop_words,
            by = "word") -> essay_tokens


## Getting counts for each word

essay_tokens %>%
  group_by(ESSAY_ID,
           word) %>%
  summarise(COUNT = n()) %>%
  ungroup() %>%
  left_join(x  = essays,
            by = "ESSAY_ID") -> essay_tokens


## Getting rid of words that appear in less than 0.5% of essays and more than 50% of essays

essay_tokens %>%
  group_by(word) %>%
  mutate(FREQ = n()) %>%
  ungroup() %>%
  mutate(FREQ = FREQ/n_distinct(ESSAY_ID)) %>%
  filter(FREQ >= 0.005,
         FREQ <= 0.5) %>%
  select(-FREQ) -> essay_tokens


## Creating the Document Term Matrix

essay_tokens %>%
  cast_dtm(term     = word,
           document = ESSAY_ID,
           value    = COUNT) -> essay_dtm



####################################################################
######## INITIAL TOPIC MODELING, PART 1 - SYSTEMATIC SEARCH ########
####################################################################

## Calculating perplexity of LDA models using a five-fold cross-validated methodology

set.seed(1)
cv_index  <- sample(1:5, nrow(essay_dtm), replace = TRUE)
topic_seq <- unique(ceiling(lseq(5, 150, 40)))

clusters  <- makeCluster(detectCores())
registerDoParallel(clusters)

clusterExport(clusters, c("essay_dtm", "topic_seq", "cv_index"))
clusterEvalQ(clusters, library(topicmodels))

foreach(i = 1:length(topic_seq)) %dopar% {
  
  LDA_scores <- data.frame()
  
  for(j in 1:5) {
    
    temp_LDA <- LDA(x       = essay_dtm[cv_index != j, ],
                    k       = topic_seq[i],
                    method  = "Gibbs",
                    control = list(burnin = 500,
                                   iter   = 500,
                                   keep   = 50,
                                   seed   = 1))
    
    temp_scores <- data.frame(topics           = topic_seq[i],
                              cv_fold          = j,
                              perplexity_train = perplexity(object  = temp_LDA, 
                                                            newdata = essay_dtm[cv_index != j, ]),
                              perplexity_test  = perplexity(object  = temp_LDA, 
                                                            newdata = essay_dtm[cv_index == j, ]))
    
    LDA_scores <- rbind(LDA_scores, temp_scores)
    
    print(paste0(topic_seq[i], "-", j))
    
    }

  write.csv(x    = LDA_scores,
            file = paste0("Initial Model Fit Statistics/LDA_", topic_seq[i], ".csv"), 
            row.names = FALSE)
  
  rm(LDA_scores)
  gc()
  
  
}
stopCluster(clusters)
rm(cv_index, topic_seq, clusters)





########################################################################
######## LOADING/CLEANING DATA - COMPILING INITIAL SEARCH STATS ########
########################################################################

## Compiling LDA models

data.frame(files = dir("Initial Model Fit Statistics")) %>%
  mutate(files = as.character(files)) %>%
  filter(str_detect(files, "^LDA_\\d") == TRUE) -> temp_files

LDA_tuning_a <- data.frame()


for (i in 1:nrow(temp_files)) {
  
  paste0("Initial Model Fit Statistics/", 
         temp_files[i, 1]) %>%
    read.csv(stringsAsFactors = FALSE) -> temp_data
  
  LDA_tuning_a <- rbind(LDA_tuning_a,
                        temp_data)
  
}
rm(i, temp_data, temp_files)



#################################################
######## DATA VISUALIZATION - LDA TUNING ########
#################################################

##

LDA_tuning %>%
  ggplot() +
    geom_point(aes(x = topics,
                   y = perplexity_test),
               alpha = 0.50) +
    geom_smooth(aes(x = topics,
                    y = perplexity_test)) +
    scale_y_continuous(name = "Perplexity",
                       breaks       = c(900, 1000, 1100, 1200, 1300, 1400),
                       minor_breaks = NULL) + 
    scale_x_continuous(name = "Number of Topics",
                       breaks       = c(0, 25, 50, 75, 100, 125, 150),
                       minor_breaks = NULL) + 
    theme(axis.ticks = element_blank())
        




#####################################################################
######## INITIAL TOPIC MODELING, PART 2 - MANUAL EXAMINATION ########
#####################################################################

## Manual exmination of LDA models

manual_LDA_topics <- c(10, 20, 25, 30, 40, 150)
clusters          <- makeCluster(detectCores())

registerDoParallel(clusters)

clusterExport(clusters, c("essay_dtm", "manual_LDA_topics"))
clusterEvalQ(clusters, c(library(topicmodels), library(tidyr)))


foreach(i = 1:length(manual_LDA_topics)) %dopar% {
  
  temp_LDA <- LDA(x       = essay_dtm,
                  k       = manual_LDA_topics[i],
                  method  = "Gibbs",
                  control = list(burnin = 500,
                                 iter   = 500,
                                 keep   = 50,
                                 seed   = 1))
  
  terms(temp_LDA, 15) %>%
    as.data.frame() %>%
    write.csv(file = paste0("Manual Examination of Models/LDA_", manual_LDA_topics[i], ".csv"),
              row.names = FALSE)

  print(manual_LDA_topics[i])
  rm(temp_LDA)
  
}
rm(i)




