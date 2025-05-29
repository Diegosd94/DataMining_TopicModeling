# Load required libraries
library(dplyr)        # Data manipulation
library(ggplot2)      # Data visualization
library(tm)           # Text mining
library(cld2)         # Language detection
library(tokenizers)   # Text tokenization
library(textstem)     # Lemmatization
library(gutenbergr)   # Gutenberg texts (if needed)
library(wordcloud)    # Wordcloud visualization
library(topicmodels)  # Topic modeling (LDA)
library(LDAvis)       # LDA visualization
library(servr)        # Interactive visualization server
library(stringr)      # String manipulation
library(RColorBrewer) # Color palettes for plots
library(readr)        # Reading CSV files
library(ldatuning)    # Finding optimal number of topics for LDA

# Load dataset
hotels_data <- read_csv("HotelsData.csv")

# Convert Review score to factor
hotels_data$`Review score` <- as.factor(hotels_data$`Review score`)

# Check column names and preview data
colnames(hotels_data)
head(hotels_data)

# Detect language of each review text
hotels_data$language <- detect_language(hotels_data$`Text 1`)

# Calculate review sentiment counts and percentages
total_positive <- nrow(filter(hotels_data, `Review score` %in% c(4, 5)))
total_negative <- nrow(filter(hotels_data, `Review score` %in% c(1, 2)))
total_neutral  <- nrow(filter(hotels_data, `Review score` == 3))
total_reviews  <- nrow(hotels_data)

positive_percentage <- (total_positive / total_reviews) * 100
negative_percentage <- (total_negative / total_reviews) * 100
neutral_percentage  <- (total_neutral  / total_reviews) * 100

summary <- data.frame(
  'Review Type' = c('Positive Reviews', 'Negative Reviews', 'Neutral Reviews'),
  'Total Reviews' = c(total_positive, total_negative, total_neutral),
  'Percentage' = c(positive_percentage, negative_percentage, neutral_percentage)
)
print(summary)

# Summary by language
language_summary <- hotels_data %>%
  group_by(language) %>%
  summarise(Review_Count = n()) %>%
  mutate(Percentage = round((Review_Count / total_reviews) * 100, 2)) %>%
  arrange(desc(Review_Count))
print(language_summary)

# Sample size proportion calculation based on Review score distribution
set.seed(105)
review_props <- hotels_data %>%
  count(`Review score`) %>%
  mutate(prop = n / sum(n),
         sample_size = round(prop * 2000))
print(review_props)

# Filter only English reviews
hotels_data_en <- filter(hotels_data, language == "en")
print(paste("Number of English reviews:", nrow(hotels_data_en)))

# Sample 2000 reviews from English reviews
set.seed(105)
sampled_data <- slice_sample(hotels_data_en, n = 2000)
print(paste("Sampled reviews count:", nrow(sampled_data)))

# Separate positive and negative samples
positive_reviews_sample <- filter(sampled_data, `Review score` %in% c(4, 5))
negative_reviews_sample <- filter(sampled_data, `Review score` %in% c(1, 2))

# --- Text Preprocessing and Topic Modeling for Positive Reviews ---

positive_texts <- positive_reviews_sample$`Text 1`
positive_corpus <- Corpus(VectorSource(positive_texts))

positive_dtm <- DocumentTermMatrix(positive_corpus, control = list(
  removePunctuation = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  tolower = TRUE
))

# Remove empty documents
positive_doc_sums <- rowSums(as.matrix(positive_dtm))
positive_dtm <- positive_dtm[positive_doc_sums != 0, ]

# Calculate term frequency and plot wordcloud
positive_term_matrix <- as.matrix(positive_dtm)
positive_term_freq <- sort(colSums(positive_term_matrix), decreasing = TRUE)

wordcloud(names(positive_term_freq)[1:200], positive_term_freq[1:100], 
          rot.per = 0.15, random.order = FALSE, scale = c(4, 0.5),
          colors = brewer.pal(8, "Dark2"))

# LDA Topic Modeling
positive_lda <- LDA(positive_dtm, k = 15, method = "Gibbs", control = list(iter = 1000, seed = 105))

positive_phi <- posterior(positive_lda)$terms
positive_theta <- posterior(positive_lda)$topics
positive_vocab <- colnames(positive_phi)

positive_json <- createJSON(phi = positive_phi, theta = positive_theta, 
                           vocab = positive_vocab, 
                           doc.length = rowSums(positive_term_matrix),
                           term.frequency = positive_term_freq)

serVis(positive_json, out.dir = 'positive_vis', open.browser = TRUE)

# --- Repeat preprocessing and modeling for Negative Reviews ---

negative_texts <- negative_reviews_sample$`Text 1`
negative_corpus <- Corpus(VectorSource(negative_texts))

negative_dtm <- DocumentTermMatrix(negative_corpus, control = list(
  removePunctuation = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  tolower = TRUE
))

negative_doc_sums <- rowSums(as.matrix(negative_dtm))
negative_dtm <- negative_dtm[negative_doc_sums != 0, ]

negative_term_matrix <- as.matrix(negative_dtm)
negative_term_freq <- sort(colSums(negative_term_matrix), decreasing = TRUE)

wordcloud(names(negative_term_freq)[1:200], negative_term_freq[1:100], 
          rot.per = 0.15, random.order = FALSE, scale = c(4, 0.5),
          colors = brewer.pal(8, "Dark2"))

negative_lda <- LDA(negative_dtm, k = 15, method = "Gibbs", control = list(iter = 1000, seed = 105))

negative_phi <- posterior(negative_lda)$terms
negative_theta <- posterior(negative_lda)$topics
negative_vocab <- colnames(negative_phi)

negative_json <- createJSON(phi = negative_phi, theta = negative_theta, 
                           vocab = negative_vocab, 
                           doc.length = rowSums(negative_term_matrix),
                           term.frequency = negative_term_freq)

serVis(negative_json, out.dir = 'negative_vis', open.browser = TRUE)

# --- Determine optimal number of topics for full sample ---

news_utf8 <- stringr::str_conv(sampled_data$`Text 1`, "UTF-8")
full_corpus <- Corpus(VectorSource(news_utf8))

full_dtm <- DocumentTermMatrix(full_corpus, control = list(
  removePunctuation = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  tolower = TRUE
))

full_doc_sums <- rowSums(as.matrix(full_dtm))
full_dtm <- full_dtm[full_doc_sums != 0, ]
full_matrix <- as.matrix(full_dtm)

topic_results <- FindTopicsNumber(
  full_matrix,
  topics = seq(3, 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
  method = "Gibbs",
  control = list(seed = 105),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(topic_results)

# --- Final LDA model on full sample ---

set.seed(105)
lda_model <- LDA(full_dtm, k = 15, method = "Gibbs", control = list(seed = 105, iter = 1000))

phi <- posterior(lda_model)$terms
theta <- posterior(lda_model)$topics

# Extract top terms per topic
top_terms <- terms(lda_model, 5)
lda_terms <- as.matrix(top_terms)
print(lda_terms)

# Create a data frame of topics and their keywords
top_terms_transposed <- t(top_terms)
topic_keywords_df <- data.frame(
  topic = paste0("Topic ", 1:nrow(top_terms_transposed)),
  keywords = apply(top_terms_transposed, 1, function(words) paste(words, collapse = ", "))
)
print(topic_keywords_df)

# Assign dominant topic to each document
dominant_topics <- data.frame(topic = topics(lda_model))
dominant_topics$doc_id <- as.numeric(rownames(dominant_topics))

# Merge topic assignment with original data
sampled_data$doc_id <- as.numeric(rownames(sampled_data))
data_with_topics <- merge(sampled_data, dominant_topics, by = "doc_id")

head(data_with_topics, 10)

# Show topic probabilities for first few documents
topic_probabilities <- as.data.frame(lda_model@gamma)
head(topic_probabilities[, 1:5])

# Calculate average review score per topic
data_with_topics$Review_score_num <- as.numeric(as.character(data_with_topics$`Review score`))
avg_review_by_topic <- data_with_topics %>%
  group_by(topic) %>%
  summarise(avg_review_score = round(mean(Review_score_num, na.rm = TRUE), 2))

topic_keywords_df$topic_num <- as.numeric(gsub("Topic ", "", topic_keywords_df$topic))
topic_keywords_df <- topic_keywords_df %>%
  left_join(avg_review_by_topic, by = c("topic_num" = "topic")) %>%
  select(topic, keywords, avg_review_score)

print(topic_keywords_df)


