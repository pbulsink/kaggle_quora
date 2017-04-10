#Model 1: similar words, and sentiments
library(syuzhet)
library(pbapply)

q1<-train$question1
q2<-train$question2

s1<-get_nrc_sentiment(q1)
s2<-get_nrc_sentiment(q2)

rm(q1, q2)

colnames(s1)<-paste("q1_", colnames(s1), sep="")
colnames(s2)<-paste("q2_", colnames(s2), sep="")

train<-cbind(train, s1, s2)

rm(s1, s2)

strDups<-function(x){
  a<-unlist(tokenize(x['question1']))
  b<-unlist(tokenize(x['question2']))
  return(sum(a%in%b)/sum(length(a), length(b)))
}

strCapDiff<-function(x){
  #returns capital words per sentence in Q1 minus capital words per sentence in Q2
  a<-unlist(tokenize(x['question1']))
  a_sent<-length(unlist(tokenize(x['question1'], what='sentence')))
  b<-unlist(tokenize(x['question2']))
  b_sent<-length(unlist(tokenize(x['question2'], what='sentence')))
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

#stopCluster(cl)

library(caret)
inTraining<-createDataPartition(train$is_duplicate, p=0.75, list=FALSE)
tr<-train[inTraining, ]
val<-train[-inTraining,]

set.seed(1)

fitControl <- trainControl(method = 'repeatedcv', number=10)

model1<-train(is_duplicate ~ ., data=tr, method="rf")
model2<-train(is_duplicate ~ ., data=tr, method='gbm')
