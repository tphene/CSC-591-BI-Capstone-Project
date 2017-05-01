#install.packages("text2vec")
#install.packages("readr")
#install.packages("glmnet")
library(text2vec)
library(readr)
library(glmnet)
require(C50)
require(caret)
require(kernlab)
require(e1071)
require(ROCR)
#predicting whether or not the text belongs to majority class i.e personal or not
train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")

#converting the fine grain data into suitable binary data for using GLM model
train$category[train$category!='personal']=as.numeric(0)
train$category[train$category=='personal']=as.numeric(1)
test$category[test$category!='personal']=as.numeric(0)
test$category[test$category=='personal']=as.numeric(1)

#converting the data predictive labels to numeric values. 
test$category = as.numeric(test$category)
train$category = as.numeric(train$category)

#using the tokenizer
prep_fun = tolower
tok_fun = word_tokenizer

#total = rbind(train,test)

it_train = itoken(train$text, preprocessor = prep_fun, tokenizer = tok_fun, ids = train$category, progressbar = FALSE)
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

#deciding on the number of folds
NFOLDS = 4

#using the GLM classifier on the above data
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['category']], 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 5-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
#plot the auc 
plot(glmnet_classifier)

it_test = test$text %>% 
  prep_fun %>% 
  tok_fun %>% 
  itoken(ids = test$category, 
         # turn off progressbar because it won't look nice in rmd
         progressbar = FALSE)

#Creating a Document term matrix for the data
dtm_test = create_dtm(it_test, vectorizer)

#generating predictions and comparing the actual results of the test set
preds = predict(glmnet_classifier, dtm_test, type = 'response')

#calculating the accuracy
accuracy = glmnet:::auc(test$category, preds)
accuracy
###########################################################################
#                               pruning vocabulary                        #
###########################################################################

#pruning the standard stop words from the text
stop_words = c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours","the")
vocab = create_vocabulary(it_train, stopwords = stop_words)

#Pruned vocabulary with min number of terms and doc proportions
pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
#Creating a vectorizer using the library
vectorizer = vocab_vectorizer(pruned_vocab)
dtm_train  = create_dtm(it_train, vectorizer)

vocab = create_vocabulary(it_train, ngram = c(1L, 2L))

vocab = vocab %>% prune_vocabulary(term_count_min = 10, 
                                   doc_proportion_max = 0.5)

bigram_vectorizer = vocab_vectorizer(vocab)
#Create a document term matrix
dtm_train = create_dtm(it_train, bigram_vectorizer)
#Using the GLM classifier on this data
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['category']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)
#plot the auc 
plot(glmnet_classifier)

dtm_test = create_dtm(it_test, bigram_vectorizer)


#generating predictions and comparing the actual results of the test set
preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]

#calculating the accuracy
accuracy = glmnet:::auc(test$category, preds)
accuracy

###########################################################################
###########################################################################
#                               using tfidf                               #
###########################################################################

#Normalize the data using l1 normalization technique
dtm_train_l1_norm = normalize(dtm_train, "l1")

vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf  = create_dtm(it_test, vectorizer) %>% 
  transform(tfidf)
#Using the GLM classifier on this data
glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train[['category']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

#plot the auc 
plot(glmnet_classifier)
#generating predictions and comparing the actual results of the test set
preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]

#calculating the accuracy
accuracy = glmnet:::auc(test$category, preds)
accuracy
