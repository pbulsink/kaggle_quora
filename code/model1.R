#Model 1: similar words, and sentiments
library(syuzhet)

q1<-train$question1
q2<-train$question2

s1<-get_nrc_sentiment(q1)
s2<-get_nrc_sentiment(q2)

colnames(s1)<-paste("q1.", colnames(s1), sep="")
colnames(s2)<-paste("q2.", colnames(s2), sep="")

cbind(train, s1, s2)

similarwordsprop<-sum(!is.na(match(tokenize(q1), tokenize(q2))), !is.na(match(tokenize(q2), tokenize(q1))))/sum(length(tokenize(q1)), tokenize(q2))

tr_index<-sample(1:nrow(train), size = 0.7*nrow(train), replace = FALSE)
tr<-train[tr_index,]
val<-train[!tr_index,]


library(caret)

#model1<-
