library(wordcloud)
library(tm)
library(SnowballC)

set.seed(123)

setwd("/home/sfrenk/Documents/Projects/grant-text-mining")

## DATA COLLECTION/PROCESSING

# Fetch data
corpus <- Corpus(DirSource("./data/sample"))

# Remove stopwords
corpus <- tm_map(corpus, removeWords, stopwords("english"))

# Eliminate extra whitespace
corpus <- tm_map(corpus, stripWhitespace)

# Convert text to lower case
corpus <- tm_map(corpus, content_transformer(tolower))

# Remove numbers
corpus <- tm_map(corpus, removeNumbers)

# Remove punctuation
corpus <- tm_map(corpus, removePunctuation)

# Stemming (collapse words to root eg learning -> learn)
corpus <- tm_map(corpus, stemDocument)

# Make term-document matrix
tdm <- TermDocumentMatrix(corpus)

## ANALYSIS

m <- as.matrix(tdm)
counts <- sort(rowSums(m), decreasing = TRUE)
count_table <- data.frame(word = names(counts), freq = counts)

# Word cloud to visualize most frequent words
wordcloud(words = count_table$word, freq = count_table$freq, min.freq = 1, max.words = 200, random.order = FALSE, colors = brewer.pal(8, "Dark2"))

# Pairwise plots of word usage for all years
m <- m[order(rowSums(m), decreasing = TRUE),]

pairs(m[1:1000,])
pairs(m[1000:2000,])
