library(quanteda)
library(SnowballC)
library(parallel)
library(pbapply)

buildSubmission<-function(predict_vals, comment, commentfile="./submissions/submissions.txt"){
  submission<-data.frame(test_id=c(0:2345795), is_duplicate = predict_vals)
  dt<-unlist(strsplit(as.character(Sys.time()), ' '))
  dt[2]<-paste(unlist(strsplit(dt[2], ':')), collapse="-")
  filename<-paste0("./submissions/submisison-", paste(dt, collapse='-'), ".csv")
  write.csv(submission, file=filename, row.names = FALSE, col.names = TRUE)
  comment<-paste(filename, comment, sep = ": ")
  cat(comment, file=commentfile, append=TRUE, sep="\n")
}

logLoss = function(pred, actual){
  -1*mean(log(pred[model.matrix(~ actual + 0) - pred > 0]))
}

#Kaggle Prep
train<-read.csv("./data/train.csv")
test<-read.csv("./data/test.csv")

train$id<-as.integer(train$id)
train$qid1<-as.integer(train$qid1)
train$qid2<-as.integer(train$qid2)
train$is_duplicate<-as.integer(train$is_duplicate)
train$question1<-as.character(train$question1)
train$question2<-as.character(train$question2)
test$question1<-as.character(test$question1)
test$question2<-as.character(test$question2)


retoken<-function(x){
  x <- gsub("\\b[0-9]+\\b", "<NUM>", x)
  x <- paste(wordStem(unlist(removeFeatures(tokenize(x, what='fasterword', removePunct=TRUE), stopwords('english'))), 'english'), collapse = " ")
  x
}

cl<-makeCluster(detectCores()-1)
clusterExport(cl, c('retoken', 'unlist','removeFeatures','tokenize','tolower','stopwords', 'paste','gsub', 'wordStem'))

train$question1b <- pbsapply(train$question1, retoken, cl=cl)
train$question2b <- pbsapply(train$question2, retoken, cl=cl)
test$question1b <- pbsapply(test$question1, retoken, cl=cl)
test$question2b <- pbsapply(test$question2, retoken, cl=cl)

stopCluster(cl)
