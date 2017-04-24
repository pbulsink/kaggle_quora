#Model 1: similar words, and sentiments
library(syuzhet)
library(pbapply)
library(quanteda)

#try vectorize
strDups<-function(x){
  #Handle stopword 0'd stuff.
  if(nchar(x['question1b']) == 0 || nchar(x['question2b']) == 0) return (0)
  a<-unlist(tokenize(x['question1b']))
  b<-unlist(tokenize(x['question2b']))
  return(sum(a%in%b)/sum(length(a), length(b)))
}

strCapDiff<-function(x){
  #returns capital words per sentence in Q1 minus capital words per sentence in Q2
  if(nchar(x['question1b']) == 0 || nchar(x['question2b']) == 0) return (0)
  a<-unlist(tokenize(x['question1b']))
  a_sent<-length(unlist(tokenize(x['question1b'], what='sentence')))
  b<-unlist(tokenize(x['question2b']))
  b_sent<-length(unlist(tokenize(x['question2b'], what='sentence')))
  a<-a[!(a %in% b)]
  b<-b[!(b %in% a)]
  a_cap<-sum(grepl("[A-Z][a-zA-Z]+", a)) 
  b_cap<-sum(grepl("[A-Z][a-zA-Z]+", b))
  return((a_cap/a_sent)-(b_cap/b_sent))
}

cosineSim<-function(x){
  corp<-corpus(c(x['question1'], x['question2']))
  dfmatrix<-dfm(corp, remove=stopwords('english'), stem=TRUE, removePunct = TRUE)
  simil<-textstat_simil(x = dfmatrix, margin = "documents", n = NULL, method = "cosine")
  simil<-as.numeric(simil)[1]
  ifelse(!is.nan(simil), return(as.numeric(simil)[1]), return(0))
}

cl<-makeCluster(detectCores()-1)
clusterExport(cl, c('strDups', 'tokenize','grepl', 'strCapDiff','cosineSim','dfm','textstat_simil','corpus', 'stopwords'))

# pbapply not parallelized? Issue#24 https://github.com/psolymos/pbapply/issues/24 
# Fixed PR25 THANKS!

#get_sentiment parallelized in personal pbulsink/suzyhet
trg$sq1<-get_sentiment(trg$question1, cl=cl)
trg$sq2<-get_sentiment(trg$question2, cl=cl)
test$sq1 <- get_sentiment(test$question1, cl=cl)
test$sq2 <- get_sentiment(test$question2, cl=cl)
gc(verbose=FALSE)

trg$dup_prop<-pbapply(trg, 1, function(x) strDups(x), cl=cl)
trg$cap_diff<-pbapply(trg, 1, function(x) strCapDiff(x), cl=cl)
trg$cos_simil<-pbapply(trg, 1, function(x) cosineSim(x), cl=cl)
gc(verbose = FALSE)
test$dup_prop<-pbapply(test, 1, function(x) strDups(x), cl=cl)
test$cap_diff<-pbapply(test, 1, function(x) strCapDiff(x), cl=cl)
test$cos_simil<-pbapply(test, 1, function(x) cosineSim(x), cl=cl)

stopCluster(cl)
gc(verbose = FALSE)

trg$question1b<-NULL
trg$question2b<-NULL
trg$qid1<-NULL
trg$qid2<-NULL

test$question1b <- NULL
test$question2b <- NULL
test$qid1 <- NULL
test$qid2 <- NULL

trg$is_duplicate<-as.factor(trg$is_duplicate)

gc(verbose = FALSE)
library(caret)
inTraining<-createDataPartition(trg$is_duplicate, p=0.70, list=FALSE)
tr<-trg[inTraining, ]
val<-trg[-inTraining,]

set.seed(1)

fitControl <- trainControl(method = 'repeatedcv', number=10, repeats = 10)
#model1<-train(is_duplicate ~ ., data=val, method="rf", na.action = na.omit)
model_gbm<-train(is_duplicate ~ sq1+sq2+dup_prop+cap_diff+cos_simil, data=tr, method='gbm', trControl = fitControl, na.action = na.omit, verbose=FALSE)
#model_rf<-train(is_duplicate ~ sq1+sq2+dup_prop+cap_diff, data=tr, method='rf', trControl = fitControl, na.action = na.omit, verbose=FALSE)
model_rda<-train(is_duplicate ~ sq1+sq2+dup_prop+cap_diff+cos_simil, data=tr, method='rda', trControl = fitControl, na.action = na.omit, verbose=FALSE)
model_lda<-train(is_duplicate ~ sq1+sq2+dup_prop+cap_diff+cos_simil, data=tr, method='lda', trControl = fitControl, na.action = na.omit, verbose=FALSE)

val$gbm_pred<-predict(model_gbm, newdata=val[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]
val$rda_pred<-predict(model_rda, newdata=val[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]
#val$rf_pred<-predict(model_rf, newdata=val[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]
val$lda_pred<-predict(model_lda, newdata=val[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]


#GBM Logloss
logLoss(pred = val$gbm_pred, actual = as.numeric(val$is_duplicate))

#RDA Logloss
logLoss(pred = val$rda_pred, actual = as.numeric(val$is_duplicate))

#RF Logloss
#logLoss(pred = val$rf_pred, actual = as.numeric(val$is_duplicate))

#LDA Logloss
logLoss(pred = val$lda_pred, actual = as.numeric(val$is_duplicate))

#generate predictions
gbm_test<-predict(model_gbm, newdata=test[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]
rda_test<-predict(model_rda, newdata=test[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]
#rf_test<-predict(model_rf, newdata=test[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]
lda_test<-predict(model_lda, newdata=test[,c("sq1","sq2","dup_prop","cap_diff", "cos_simil")], type='prob', na.action = na.exclude)[,2]

gbm_test2<-ifelse(gbm_test>=0.475, 1, 0)
rda_test2<-ifelse(rda_test>=0.425, 1, 0)
lda_test2<-ifelse(lda_test>=0.425, 1, 0)

#write prediction files
buildSubmission(gbm_test2, "GBM modeled w/ repeatedcv including cosine similarity, string comparison, sentiment, and capital letter adjustments. Adjusted to 0/1 at 0.475% prediction")
buildSubmission(rda_test2, "RDA modeled w/ repeatedcv including cosine similarity, string comparison, sentiment, and capital letter adjustments. Adjusted to 0/1 at 0.425% prediction")
#buildSubmission(rf_test, "RF modeled w/ repeatedcv including cosine similarity, string comparison, sentiment, and capital letter adjustments")
buildSubmission(lda_test2, "LDA modeled w/ repeatedcv including cosine similarity, string comparison, sentiment, and capital letter adjustments. Adjusted to 0/1 at 0.425% prediction")