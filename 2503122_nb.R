# =====================================================================
# CS230/561 Project: Binary Naive Bayes (SMS spam)
# Name: J Sai Shreyan          Roll No.: 2503122
# Save this file as <rollno>_nb.R and submit it (single .R file).
# =====================================================================
#
# INSTRUCTIONS (do not delete)
# - Keep all function names, arguments and object names unchanged:
#     preprocess_text, build_vocab, compute_priors, compute_likelihoods,
#     score_message, predict_nb, tokens, labels, train_idx, test_idx,
#     acc, cm, example_preds, top_spam, top_ham
#   They are auto-graded.
# - Do not change set.seed(230) or the 80/20 split code.
# - Use alpha = 1 unless told otherwise.
# - Put SMSSpamCollection in the same folder as this file. Use R 4.x so the
#   split is reproducible.
# - Write your own code; do not use packages that implement Naive Bayes
#   (for example e1071, naivebayes).
# - The script must run top to bottom without errors:
#     source("<rollno>_nb.R")
# - Explanations, results and the summary are written as comments.
#
# GRADING RUBRIC (60 points)
#   1. Data preprocessing
#        Text cleaning and tokenization (preprocess_text) ........ 10
#        Train/test split (unchanged seed, 80/20, disjoint) ....... 5
#   2. Binary Naive Bayes implementation
#        Prior computation (compute_priors) ....................... 5
#        Likelihood estimation (compute_likelihoods) .............. 10
#        Log-posterior scoring (score_message, predict_nb) ........ 10
#   3. Evaluation and results
#        Accuracy (acc) ........................................... 5
#        Confusion matrix (cm, 2x2, spam/ham) ..................... 5
#        SMS test cases (example_preds) ........................... 5
#   4. Analysis and summary
#        Top-10 words (top_spam, top_ham) and discussion .......... 5
#   Total ........................................................ 60
# =====================================================================


# =====================================================================
# 1. Data loading and preprocessing
# =====================================================================
# Preprocessing rule (follow exactly):
#   1. Replace non-ASCII characters with a space.
#   2. Convert to lowercase.
#   3. Delete punctuation (replace with the empty string, not a space).
#   4. Split on whitespace and drop empty tokens.
# Examples: "Win FREE tickets now!!!" -> "win" "free" "tickets" "now"
#           "don't" -> "dont"      "rate)T&C's" -> "ratetcs"

# Input: one SMS string. Output: character vector of tokens (rule above).
preprocess_text <- function(text) {
  # TODO
  text <- iconv(text,from = "UTF-8",to = "ASCII",sub = " ")
  text <- tolower(text)
  text <- gsub("[[:punct:]]","", text)
  temp <- unlist(strsplit(text,"\\s+"))
  temp <- temp[temp != ""]
  return(temp)
}

# The file is tab-separated with no header: label <TAB> text.
# quote = "" is required (messages contain ' and " characters).
sms <- read.delim("SMSSpamCollection", header = FALSE, sep = "\t",
                  col.names = c("label", "text"), quote = "",
                  stringsAsFactors = FALSE)
stopifnot(nrow(sms) == 5574, all(sms$label %in% c("ham", "spam")))
print(table(sms$label))   # expect ham 4827, spam 747

tokens <- lapply(sms$text, preprocess_text)   # list of character vectors
labels <- sms$label                           # "ham" / "spam"

set.seed(230)                                 # DO NOT CHANGE
train_idx <- sample(seq_along(labels), size = floor(0.8 * length(labels)))  # DO NOT CHANGE
test_idx  <- setdiff(seq_along(labels), train_idx)

# EXPLANATION (TODO, 1-2 sentences): describe your preprocessing choices.
# Built-in functions of R are used to preprocess the input data.
# iconv to replace non-ASCII characters with a space, tolower() to convert to lowercase, gsub() to replace punctuation with a blank, unlist and strsplit to split into tokens.


# =====================================================================
# 2. Binary Naive Bayes (from scratch)
# =====================================================================

# Vocabulary: unique tokens in the TRAINING documents only (character vector).
build_vocab <- function(train_tokens) {
  # TODO
  train_tokens <- unique(unlist(train_tokens))
  return(train_tokens)
}

# Named numeric vector of P(C); names = class labels ("ham", "spam").
compute_priors <- function(train_labels) {
  # TODO
  counts <- table(train_labels)
  priors <- as.numeric(counts)/length(train_labels)
  names(priors) <- names(counts)
  return (priors)
}

# Matrix |V| x |classes|; rownames = vocab, colnames = classes.
# Entry = (#docs in class containing w + alpha) / (#docs in class + 2*alpha)
# Count a word at most once per document (presence, not frequency).
compute_likelihoods <- function(train_tokens, train_labels, vocab, alpha = 1) {
  # TODO
  classes <- names(table(train_labels))
  likelihoods <- matrix(0,nrow = length(vocab),ncol = 2,dimnames = list(vocab,classes))
  
  for (cls in classes) {
    class_docs <- train_tokens[train_labels == cls]
    n_docs <- length(class_docs)
    
    unique_per_doc <- unlist(lapply(class_docs, unique))
    counts <- table(unique_per_doc)[vocab]
    counts[is.na(counts)] <- 0
    
    likelihoods[, cls] <- (as.numeric(counts) + alpha) / (n_docs + 2 * alpha)
  }
  
  return(likelihoods)
}

# Named numeric vector of log-scores, one per class:
#   log P(C) + sum over w in V of [ I(w) log P(w|C) + (1 - I(w)) log(1 - P(w|C)) ]
# Words not in the vocabulary are ignored.
score_message <- function(msg_tokens, priors, lik, vocab) {
  # TODO
  present_words <- intersect(unique(msg_tokens), vocab)
  absent_words  <- setdiff(vocab, present_words)
  
  scores <- numeric(length(priors))
  names(scores) <- names(priors)
  
  for (cls in names(scores)) {
    scores[cls] <- log(priors[cls]) + sum(log(lik[present_words, cls])) + sum(log(1 - lik[absent_words, cls]))
  }
  
  return(scores)
}

# Character vector of predicted classes (argmax of score_message), one per message.
predict_nb <- function(token_list, priors, lik, vocab) {
  # TODO
  predictions <- character(length(token_list))
  
  for(i in seq_along((token_list))){
    scores <- score_message(token_list[[i]], priors, lik, vocab)
    predictions[i] <- names(scores)[which.max(scores)]
  }
  
  return(predictions)
}


# =====================================================================
# 3. Training and evaluation
# =====================================================================
vocab  <- build_vocab(tokens[train_idx])
priors <- compute_priors(labels[train_idx])
lik    <- compute_likelihoods(tokens[train_idx], labels[train_idx], vocab, alpha = 1)

test_pred <- predict_nb(tokens[test_idx], priors, lik, vocab)

acc <- mean(test_pred == labels[test_idx])   # TODO: accuracy on the test set (a single number)
cm  <- table(Actual = factor(labels[test_idx], levels = c("ham","spam")), Predicted = factor(test_pred , levels = c("ham","spam")))   # TODO: 2x2 table; rows = actual, columns = predicted, e.g.
#   table(Actual    = factor(labels[test_idx], levels = c("ham", "spam")),
#         Predicted = factor(test_pred,        levels = c("ham", "spam")))
print(acc); print(cm)

example_msgs <- c("Win free tickets now!!!", "Are you coming to the meeting?",
                  "URGENT! You won $1000", "See you tomorrow at lunch")
example_tokens <- lapply(example_msgs,preprocess_text)
example_preds <- predict_nb(example_tokens,priors,lik,vocab)  # TODO: character vector of 4 predictions (preprocess, then predict_nb)
print(data.frame(message = example_msgs, predicted = example_preds))

# RESULTS (TODO): in 2-3 comment lines, report the accuracy, say what the
# confusion matrix shows (false positives vs false negatives), and whether the
# four example messages were classified as expected.
# accuracy reported is 97.04%

#CONFUSION MATRIX :
#                Predicted-ham    Predicted-spam
# Actual-ham     960              0
# Actual-spam    33               122
# Two out of four messages are classified as expected. Messages 1 and 3 were predicted as ham, but they were actually spam. This points out the limitations of the Naive Bayes assumption.

# =====================================================================
# 4. Analysis and summary
# =====================================================================
# Ranking rule: rank words by the log-odds
#   log P(w | spam) - log P(w | ham)   computed from `lik`.
# top_spam = the 10 largest values; top_ham = the 10 smallest (most negative).
# Both must be named numeric vectors (names = words).
log_odds <- log(lik[,"spam"]) - log(lik[,"ham"])
sorted_odds <- sort(log_odds, decreasing = TRUE)
top_spam <- head(sorted_odds,10)  # TODO
top_ham  <- head(sort(log_odds, decreasing = FALSE), 10)  # TODO
print(top_spam); print(top_ham)

# DISCUSSION (TODO, 4-6 sentences, as comments):
# - What kinds of words indicate spam, and what kinds indicate ham?
# - Name one limitation of the Naive Bayes independence assumption, with an
#   example from your results.
# - Name one thing you would change to improve the classifier.
# 
# Words like claim, prize, won, guaranteed and numbers like 1000, 500 indicate spam. While words like lor, later, he, da, ill, she indicate ham.
# Naive Bayes assumption is based on independence of words/tokens. It basically assumes words are independent in a sentence, but that is totally false in basic English. Very often, words throughout the sentence are interconnected.
# For example, "free" has a completely different meaning in "Are you free tonight?" and "You got $1000 for free". The same word acts with a different meaning in different cases.
# I would adjust the smoothing value alpha to a suitable value so that accuracy is maximum; we can achieve this by trial and error with different values of alpha and testing on the same data set.
