library("dplyr")
library(ggplot2)
#install.packages("cld2")
library(tm)
library(cld2)

library(tokenizers)
library(textstem)
library(gutenbergr)
#install.packages("wordcloud")
library(topicmodels)
library(LDAvis)

library(wordcloud)

library(readr)


library(dplyr) # basic data manipulation
library(tm) # package for text mining package
library(stringr) # package for dealing with strings
library(RColorBrewer)# package to get special theme color
library(wordcloud) # package to create wordcloud
library(topicmodels) # package for topic modelling
library(ggplot2) # basic data visualization
library(LDAvis) # LDA specific visualization 
library(servr) # interactive support for LDA visualization

hotels_data <- read_csv("HotelsData.csv")
colnames(hotels_data)

head(hotels_data)
hotels_data$language <- detect_language(hotels_data$`Text 1`)

set.seed(105)

positive_reviews <- hotels_data %>%
  filter(`Review score` %in% c(4, 5))

negative_reviews <- hotels_data %>%
  filter(`Review score` %in% c(1, 2))

neutral_reviews <- hotels_data %>%
  filter(`Review score` == 3)


total_positive <- nrow(positive_reviews)
total_negative <- nrow(negative_reviews)
total_neutral <- nrow(neutral_reviews)
total_reviews <- nrow(hotels_data)


language_summary <- hotels_data %>%
  group_by(language) %>%
  summarise(Review_Count = n()) %>%
  mutate(Percentage = round((Review_Count / total_reviews) * 100, 2)) %>%
  arrange(desc(Review_Count))
print(language_summary)


positive_reviews_sample <- positive_reviews %>%
  sample_n(1000)  %>% filter(language == "en")
  
negative_reviews_sample <- negative_reviews %>%
  sample_n(1000)  %>% filter(language == "en")



total_positive <- nrow(positive_reviews)
total_negative <- nrow(negative_reviews)
total_neutral <- nrow(neutral_reviews)
total_reviews <- nrow(hotels_data)

positive_percentage <- (total_positive / total_reviews) * 100
negative_percentage <- (total_negative / total_reviews) * 100
neutral_percentage <- (total_neutral / total_reviews) * 100

summary <- data.frame(
  'Review Type' = c('Positive Reviews', 'Negative Reviews', 'Neutral Reviews'),
  'Total Reviews' = c(total_positive, total_negative, total_neutral),
  'Percentage' = c(positive_percentage, negative_percentage, neutral_percentage)
)
summary



# Pre process for positiv review
positive_reviews_sample_text <- positive_reviews_sample$`Text 1` 

#  Corpus
positive_docs <- Corpus(VectorSource(positive_reviews_sample_text))

# DTM
positive_dtmdocs <- DocumentTermMatrix(positive_docs, control = list(
  lemma = TRUE, 
  removePunctuation = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  tolower = TRUE
))

#  deletate empty docs
positive_raw_sum <- apply(positive_dtmdocs, 1, FUN = sum)
positive_dtmdocs <- positive_dtmdocs[positive_raw_sum != 0,]

# matrix
positive_dtm_new <- as.matrix(positive_dtmdocs)

# frequency of terms
positive_frequency <- colSums(positive_dtm_new)
positive_frequency <- sort(positive_frequency, decreasing = TRUE)
positive_frequency
# worldcloud 
positive_words <- names(positive_frequency)
wordcloud(positive_words[1:200], positive_frequency[1:200], rot.per = 0.15, 
          random.order = FALSE, scale = c(4, 0.5),
          random.color = FALSE, colors = brewer.pal(8, "Dark2"))


# LDA
positive_lda <- LDA(positive_dtmdocs, k = 3, method = "Gibbs", control = list(iter = 1000, seed = 105))


positive_phi <- posterior(positive_lda)$terms %>% as.matrix()  
positive_theta <- posterior(positive_lda)$topics %>% as.matrix()  


positive_vocab <- colnames(positive_phi) 
positive_json_lda <- createJSON(phi = positive_phi, theta = positive_theta, 
                                vocab = positive_vocab, doc.length = rowSums(positive_dtm_new),
                                term.frequency = positive_frequency)

serVis(positive_json_lda, out.dir = 'positive_vis', open.browser = TRUE)




# Pre negative review
negative_reviews_sample_text <- negative_reviews_sample$`Text 1`  
head(negative_reviews_sample$`Text 1`)
#  Corpus
negative_docs <- Corpus(VectorSource(negative_reviews_sample_text))

# DTM
negative_dtmdocs <- DocumentTermMatrix(negative_docs, control = list(
  lemma = TRUE, 
  removePunctuation = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  tolower = TRUE
))

# delete empty docs
negative_raw_sum <- apply(negative_dtmdocs, 1, FUN = sum)
negative_dtmdocs <- negative_dtmdocs[negative_raw_sum != 0,]

# matrix
negative_dtm_new <- as.matrix(negative_dtmdocs)

# freq terms
negative_frequency <- colSums(negative_dtm_new)
negative_frequency <- sort(negative_frequency, decreasing = TRUE)


negative_words <- names(negative_frequency)
wordcloud(negative_words[1:200], negative_frequency[1:200], rot.per = 0.15, 
          random.order = FALSE, scale = c(4, 0.5),
          random.color = FALSE, colors = brewer.pal(8, "Dark2"))


# LDA
negative_lda <- LDA(negative_dtmdocs, k =3, method = "Gibbs", control = list(iter = 1000, seed = 105))


negative_phi <- posterior(negative_lda)$terms %>% as.matrix()  
negative_theta <- posterior(negative_lda)$topics %>% as.matrix()


negative_vocab <- colnames(negative_phi)  
negative_json_lda <- createJSON(phi = negative_phi, theta = negative_theta, 
                                vocab = negative_vocab, doc.length = rowSums(negative_dtm_new),
                                term.frequency = negative_frequency)

serVis(negative_json_lda, out.dir = 'negative_vis', open.browser = TRUE)

