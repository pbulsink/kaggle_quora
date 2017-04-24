#From https://www.kaggle.com/brandenkmurray/quora-question-pairs/h2o-word2vec-starter/code
# This is an example of H2O's recently added word2vec model which is loosely based
# on the script found here: https://github.com/h2oai/h2o-3/blob/master/h2o-r/demos/rdemo.word2vec.craigslistjobtitles.R
library(data.table)
library(h2o)
h2o.init(nthreads = -1)

tr1 <- fread("./data/train.csv", select=c("id","question1","question2","is_duplicate"))
ts1 <- fread("./data/test.csv", select=c("test_id","question1","question2"))

print("Some question cleanup")
# It is important to remove "\n" -- it appears to cause a parsing error when converting to an H2OFrame
tr1[,":="(question1=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question1),
          question2=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question2))]
tr1[,":="(question1=gsub("  ", " ", question1),
          question2=gsub("  ", " ", question2))]
ts1[,":="(question1=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question1),
          question2=gsub("'|\"|'|“|”|\"|\n|,|\\.|…|\\?|\\+|\\-|\\/|\\=|\\(|\\)|‘", "", question2))]
ts1[,":="(question1=gsub("  ", " ", question1),
          question2=gsub("  ", " ", question2))]


print("get list of unique questions")
# Using only questions from the training set because the test set has 'questions' that are fake
questions <- as.data.table(rbind(tr1[,.(question=question1)], tr1[,.(question=question2)], ts1[,.(question=question1)], ts1[,.(question=question2)]))
questions <- unique(questions)
questions.hex <- as.h2o(questions, destination_frame = "questions.hex", col.types=c("String"))

STOP_WORDS = c("ax","i","you","edu","s","t","m","subject","can","lines","re","what",
               "there","all","we","one","the","a","an","of","or","in","for","by","on",
               "but","is","in","a","not","with","as","was","if","they","are","this","and","it","have",
               "from","at","my","be","by","not","that","to","from","com","org","like","likes","so")

tokenize <- function(sentences, stop.words = STOP_WORDS) {
  tokenized <- h2o.tokenize(sentences, "\\\\W+")
  
  # convert to lower case
  tokenized.lower <- h2o.tolower(tokenized)
  # remove short words (less than 2 characters)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths >= 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.lower[h2o.grep("[0-9]", tokenized.lower, invert = TRUE, output.logical = TRUE),]
  
  # remove stop words
  tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% STOP_WORDS),]
}

print("Break questions into sequence of words")
words <- tokenize(questions.hex$question)


print("Build word2vec model")
vectors <- 10 # Only 10 vectors to save time & memory
w2v.model <- h2o.word2vec(words
                          , model_id = "w2v_model"
                          , vec_size = vectors
                          , min_word_freq = 5
                          , window_size = 5
                          , init_learning_rate = 0.025
                          , sent_sample_rate = 0
                          , epochs = 5) # only a one epoch to save time

h2o.rm('questions.hex') # no longer needed

print("Sanity check - find synonyms for the word 'water'")
print(h2o.findSynonyms(w2v.model, "water", count = 5))

print("Get vectors for each question")
question_all.vecs <- h2o.transform(w2v.model, words, aggregate_method = "AVERAGE")

print("Convert to data.table & merge results")
# Could do the rest of these steps in H2O but I'm a data.table addict
question_all.vecs <- as.data.table(question_all.vecs)
questions_all <- cbind(questions, question_all.vecs)
tr1 <- merge(tr1, questions_all, by.x="question1", by.y="question", all.x=TRUE, sort=FALSE)
tr1 <- merge(tr1, questions_all, by.x="question2", by.y="question", all.x=TRUE, sort=FALSE)
ts1 <- merge(ts1, questions_all, by.x="question1", by.y="question", all.x=TRUE, sort=FALSE)
ts1 <- merge(ts1, questions_all, by.x="question2", by.y="question", all.x=TRUE, sort=FALSE)
colnames(tr1)[5:ncol(tr1)] <- c(paste0("q1_vec_C", 1:vectors), paste0("q2_vec_C", 1:vectors))
colnames(ts1)[4:ncol(ts1)] <- c(paste0("q1_vec_C", 1:vectors), paste0("q2_vec_C", 1:vectors))

tr1$is_duplicate<-as.factor(tr1$is_duplicate)

set.seed(1)
in_training<-sample(1:nrow(tr1), size = 0.8*nrow(tr1), replace = FALSE)
trg<-as.h2o(tr1[in_training,])
val<-as.h2o(tr1[!in_training,])

dep<-colnames(tr1)[5:24]

#training
nfolds <- 5

model2.gbm<-h2o.gbm(x=dep, y="is_duplicate", training_frame = trg, validation_frame = val, nfolds=nfolds, fold_assignment = "Modulo", keep_cross_validation_fold_assignment = TRUE, keep_cross_validation_predictions = TRUE, seed=1)
#model2.glm<-h2o.glm(x=dep, y="is_duplicate", training_frame = trg, validation_frame = val, nfolds=nfolds, fold_assignment = "Modulo", keep_cross_validation_fold_assignment = TRUE, keep_cross_validation_predictions = TRUE, seed=1)
model2.nb<-h2o.naiveBayes(x=dep, y="is_duplicate", training_frame = trg, validation_frame = val, nfolds=nfolds, fold_assignment = "Modulo", keep_cross_validation_fold_assignment = TRUE, keep_cross_validation_predictions = TRUE, seed=1)
model2.rf<-h2o.randomForest(x=dep, y="is_duplicate", training_frame = trg, validation_frame = val, nfolds=nfolds, fold_assignment = "Modulo", keep_cross_validation_fold_assignment = TRUE, keep_cross_validation_predictions = TRUE, seed=1)
model2.stack<-h2o.stackedEnsemble(x=dep, y="is_duplicate", training_frame = trg, validation_frame = val, base_models = list(model2.gbm@model_id, model2.rf@model_id, model2.nb@model_id))

test_p<-h2o.predict(model2, as.h2o(ts1[,..dep]))