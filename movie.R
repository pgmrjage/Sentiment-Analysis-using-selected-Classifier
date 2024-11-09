# Set working directory and load the dataset
setwd("D:\\School Workspace\\Source Codes\\R Language\\Classifier")
df <- read.csv("train_df.csv")

# Inspect the structure of the data
str(df)

# Convert 'sentiment' to a factor variable (since it's categorical)
df$sentiment <- factor(df$sentiment, levels = c("negative", "neutral", "positive"))

# Split data into training and testing datasets
df_1000 <- df[1:1000, ]  # Select first 1000 rows
train_data <- df_1000[1:800, ]
test_data <- df_1000[801:1000, ]

# Summary of 'sentiment' count in the training data
sentiment_count <- table(as.factor(train_data$sentiment))
print(sentiment_count)

# Let's say we want to predict the 'label' column based on the 'sentiment' column
# Train a linear regression model
mod <- lm(label ~ sentiment, data = train_data)

# View summary of the linear model
summary(mod)

# Plot the linear regression model (diagnostic plots)
plot(mod)



# Create a scatter plot of the 'label' against 'sentiment'
# Convert 'sentiment' to a numeric factor for plotting
train_data$sentiment_numeric <- as.numeric(train_data$sentiment)

# Plot label vs sentiment
plot(train_data$sentiment_numeric, train_data$label, 
     xlab = "Sentiment", ylab = "Label", 
     main = "Linear Regression: Sentiment vs Label", 
     pch = 19, col = "blue")

# Add the regression line
abline(mod, col = "red", lwd = 2)  # This adds the regression line

# Alternatively, using ggplot2 for a more advanced plot
library(ggplot2)
ggplot(train_data, aes(x = sentiment_numeric, y = label)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Linear Regression: Sentiment vs Label", x = "Sentiment", y = "Label")
