library(wordcloud)
library(tm)
library(SnowballC)

set.seed(123)

setwd("/home/sfrenk/Documents/Projects/grant-text-mining")

make_wordcloud <- function(year){
    
    data <- read.csv(paste0("RePORTER_PRJABS_C_FY", year, ".csv"))
    #data <- read.csv("RePORTER_PRJABS_C_FY2016.csv", stringsAsFactors = FALSE)
    
    # Sample the data to save memory
    rows <- sample(nrow(data), 1000)
    data <- data[rows,]
    
    # Create corpus
    corpus <- Corpus(VectorSource(data$ABSTRACT_TEXT))
    
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
    
    m <- as.matrix(tdm)
    counts <- sort(rowSums(m), decreasing = TRUE)
    count_table <- data.frame(word = names(counts), freq = counts)
    return(count_table)
    
    # Make word cloud
    #return(wordcloud(words = count_table$word, freq = count_table$freq, min.freq = 1, max.words = 200, random.order = FALSE, colors = brewer.pal(8, "Dark2")))

}

wc_1986 <- make_wordcloud(1986)
wc_2016 <- make_wordcloud(2016)

wc_1986 <- head(wc_1986, 1000)
wc_2016 <- head(wc_2016, 1000)

wc_both <- merge(wc_1986, wc_2016, by = "word", all = FALSE, sort = FALSE)
wc_both <- wc_both[10:nrow(wc_both),]

plot(wc_both$freq.x, wc_both$freq.y)
head(wc_both)
summary(lm(wc_both$freq.y~wc_both$freq.x))
