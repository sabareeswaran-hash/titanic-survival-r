# Week 4: Model Building and Predictive Analysis (Titanic) 
# Put titanic.csv in the same folder as this script, then run it in RStudio.
# Install once: install.packages(c('tidyverse','caret','randomForest','pROC'))
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)

set.seed(42)   # makes results reproducible

df <- read_csv("titanic.csv")

df <- df %>%
  mutate(
    Title = str_extract(Name, "(?<=, )[^.]+(?=\\.)"),
    Title = case_when(
      Title == "Mr" ~ "Mr",
      Title %in% c("Mrs", "Mme", "Ms") ~ "Mrs",
      Title %in% c("Miss", "Mlle") ~ "Miss",
      Title == "Master" ~ "Master",
      TRUE ~ "Other"),
    FamilySize = SibSp + Parch + 1
  ) %>%
  group_by(Title) %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age)) %>%
  ungroup() %>%
  mutate(
    Embarked = replace_na(Embarked, "S"),
    Survived = factor(Survived, levels = c(0, 1), labels = c("No", "Yes")),
    Pclass   = factor(Pclass),
    Sex      = factor(Sex),
    Embarked = factor(Embarked),
    Title    = factor(Title)
  ) %>%
  select(Survived, Pclass, Sex, Age, FamilySize, Fare, Embarked, Title)

glimpse(df)
prop.table(table(df$Survived))   # class balance

idx   <- createDataPartition(df$Survived, p = 0.8, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

dim(train); dim(test)
prop.table(table(train$Survived))
prop.table(table(test$Survived))

ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Model 1: Logistic regression
glm_fit <- train(Survived ~ ., data = train,
                 method = "glm", family = "binomial",
                 trControl = ctrl, metric = "ROC")

# Model 2: Random forest
rf_fit <- train(Survived ~ ., data = train,
                method = "rf",
                tuneGrid = expand.grid(mtry = c(2, 3, 4, 6)),
                ntree = 300,
                trControl = ctrl, metric = "ROC")

glm_fit
rf_fit   # shows the cross-validated ROC for each mtry value

evaluate <- function(fit, name) {
  pred <- predict(fit, test)
  prob <- predict(fit, test, type = "prob")[, "Yes"]
  cm   <- confusionMatrix(pred, test$Survived, positive = "Yes")
  r    <- roc(test$Survived, prob, levels = c("No", "Yes"), direction = "<")
  print(cm$table)
  data.frame(Model     = name,
             Accuracy  = round(cm$overall["Accuracy"], 3),
             Precision = round(cm$byClass["Precision"], 3),
             Recall    = round(cm$byClass["Recall"], 3),
             F1        = round(cm$byClass["F1"], 3),
             AUC       = round(as.numeric(auc(r)), 3),
             row.names = NULL)
}

results <- rbind(evaluate(glm_fit, "Logistic Regression"),
                 evaluate(rf_fit,  "Random Forest"))
print(results)
write.csv(results, "model_results.csv", row.names = FALSE)

roc_glm <- roc(test$Survived, predict(glm_fit, test, type = "prob")[, "Yes"],
               levels = c("No", "Yes"), direction = "<")
roc_rf  <- roc(test$Survived, predict(rf_fit, test, type = "prob")[, "Yes"],
               levels = c("No", "Yes"), direction = "<")

p_roc <- ggroc(list("Logistic Regression" = roc_glm, "Random Forest" = roc_rf)) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(title = "ROC Curves on the Test Set", colour = "Model") +
  theme_minimal()
ggsave("roc_curves.png", p_roc, width = 6, height = 4.5, dpi = 150)

imp <- varImp(rf_fit)$importance %>%
  rownames_to_column("Feature") %>%
  arrange(desc(Overall))
p_imp <- ggplot(imp, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_col(fill = "steelblue") + coord_flip() +
  labs(title = "Random Forest Variable Importance", x = NULL, y = "Importance") +
  theme_minimal()
ggsave("variable_importance.png", p_imp, width = 6, height = 4.5, dpi = 150)

summary(glm_fit$finalModel)   # coefficients of the logistic model
