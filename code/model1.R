#Model 1: similar words, and sentiments
library(syuzhet)
library(pbapply)

train$sq1<-get_sentiment(train$question1)
train$sq2<-get_sentiment(train$question2)

#try vectorize
strDups<-function(x){
  a<-unlist(tokenize(x['question1b']))
  b<-unlist(tokenize(x['question2b']))
  return(sum(a%in%b)/sum(length(a), length(b)))
}

strCapDiff<-function(x){
  #returns capital words per sentence in Q1 minus capital words per sentence in Q2
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

#cl<-makeCluster(detectCores()-1)
#clusterExport(cl, c('strDups', 'tokenize','grepl', 'strCapDiff'))

#pbapply not parallelized? Issue#24 https://github.com/psolymos/pbapply/issues/24
train$dup_prop<-pbapply(train, 1, function(x) strDups(x))#, cl=cl)
train$cap_diff<-pbapply(train, 1, function(x) strCapDiff(x))#, cl=cl)
test$dup_prop<-pbapply(test, 1, function(x) strDups(x))#, cl=cl)
test$cap_diff<-pbapply(test, 1, function(x) strCapDiff(x))#, cl=cl)

#stopCluster(cl)

#train$question1b<-NULL
#train$question2b<-NULL
train$qid1<-NULL
train$qid2<-NULL

train$is_duplicate<-as.factor(train$is_duplicate)

library(caret)
inTraining<-createDataPartition(train$is_duplicate, p=0.75, list=FALSE)
tr<-train[inTraining, ]
val<-train[-inTraining,]

set.seed(1)

fitControl <- trainControl(method = 'repeatedcv', number=10)
# 
#model1<-train(is_duplicate ~ ., data=val, method="rf", na.action = na.omit)
model_gbm<-train(is_duplicate ~ sq1+sq2+dup_prop+cap_diff, data=tr, method='gbm', trControl = fitControl, na.action = na.omit, verbose=FALSE)
model_svm<-train(is_duplicate ~ sq1+sq2+dup_prop+cap_diff, data=tr, method='svmRadial', trControl = fitControl, na.action = na.omit, verbose=FALSE)
model_rda<-train(is_duplicate ~ sq1+sq2+dup_prop+cap_diff, data=tr, method='rda', trControl = fitControl, na.action = na.omit, verbose=FALSE)
gbm_pre<-predict(object = )