setwd("D:\\School Workspace\\Source Codes\\R Language\\Classifier")
df <- read.csv("train_df.csv")

str(df)

df$sentiment <- factor(df$sentiment, levels = c("negative", "neutral", "positive")) # nolint

df_1000 <- df[1:1000, ]
train_data <- df_1000[1:800, ]
test_data <- df_1000[801:1000, ]

sentiment_count <- table(as.factor(train_data$sentiment))
print(sentiment_count)

library(randomForest)
library(e1071)
library(caret)

mod_rf <- randomForest(sentiment ~ label, data = train_data, ntree = 100)
print(mod_rf)

mod_svm <- svm(sentiment ~ label, data = train_data, kernel = "linear")
print(mod_svm)

predictions_rf <- predict(mod_rf, test_data)

predictions_svm <- predict(mod_svm, test_data)

conf_matrix_rf <- confusionMatrix(predictions_rf, test_data$sentiment)
conf_matrix_svm <- confusionMatrix(predictions_svm, test_data$sentiment)

print("Confusion Matrix for Random Forest:")
print(conf_matrix_rf)

print("Confusion Matrix for SVM:")
print(conf_matrix_svm)

precision_rf <- posPredValue(predictions_rf, test_data$sentiment, positive = "positive") # nolint
recall_rf <- sensitivity(predictions_rf, test_data$sentiment, positive = "positive") # nolint

precision_svm <- posPredValue(predictions_svm, test_data$sentiment, positive = "positive") # nolint
recall_svm <- sensitivity(predictions_svm, test_data$sentiment, positive = "positive") # nolint

cat("\nPrecision for 'positive' using Random Forest: ", precision_rf, "\n")
cat("Recall for 'positive' using Random Forest: ", recall_rf, "\n")

cat("\nPrecision for 'positive' using SVM: ", precision_svm, "\n")
cat("Recall for 'positive' using SVM: ", recall_svm, "\n")

print("Random Forest - Classification Report:")
print(conf_matrix_rf$byClass)

print("SVM - Classification Report:")
print(conf_matrix_svm$byClass)
