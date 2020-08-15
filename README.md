# Predicting-Loan-Defaults-with-Logistic-Regression
---
title: "Project Final Report"
author: "Miriam Nwaru"
date: "8/5/2019"
output:
  word_document: default
  pdf_document: default
  html_document:
    df_print: paged
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

#Predicting Loan Defaults with Logistic Regression

##Executive Summary
This report was commissioned to predict which applicants were more likely to default on their loans. Logistic regression was used to build a model that would help carry out the predictions. Using the model without any modifications resulted in an accuracy of about 79%. About 12% of the bad loans were correctly predicted as bad, and about 97.4% of the good loans were correctly predicted as good.

To help optimize the accuracy of the model and the amount of profit it could help gain, different thresholds (the proportion needed to classify a "Good" or "Bad" loan) were tested. A threshold of .5 yielded the highest overall accuracy at 79.23%, while .7 yielded the highest profit at $3,870,539. 

It is recommended that the model used by the bank uses the .5 threshold that will yield the highest overall accuracy and also yield a profit of $2,410,333. With the model getting better over time, this will allow the bank to take more applicants, using the model to pick good loans applicants out from the pack.

The report also notes that the model does come with limitations.Since the highest accurate rate and the maximum profit from the model does not share the same threshold, our model will never be able to produce the highest accurate and maxiumum profit at the same time. Also any incorrect computations done to create the model will cause results from the model to become innaccurate.
##Setup

```{r echo=FALSE, warning = FALSE, message=FALSE}
library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(caret)
```
##Introduction:
The dataset obtained from datascienceuwl (https://datascienceuwl.github.io/Project2018/TheData.html) contains information about loans. There are 50,000 loans that were sampled and includes 30 different variables. The dataset will be preapared and cleaned, removing unneeded variables and transforming variables into a useable format. 
The new dataset will be used to predict which applicants will default on their loans. Logistic Regression will be used.

##Preparing and Cleaning the Data:
```{r}
loans = read.csv('loans50k.csv',  header = T)
```
A new variable "response" was created to help categorize the "status" variable. The "response" variable is a factor variable that has two levels: "Good", which are the loans that are fully paid, and "Bad", which are loans that have a status of charged off or default. All the other statuses that did not fall into a response category were converted to "NA".

New variables "risk", "employment_length", and "region" were also created to help categorize their respective variable. Categorizing these variables helps to easily use these variables as predictors. To help categorize "grade", a new variable "risk" was created. A grade of A or B showed a loan was the least risky compared to others. These were categorized as "Least Risk" in the "risk" variable. A grade of C or D showed a loan had a medium risk. These were categorized as "Medium Risk". A grade of F or G showed a loan had the most risk. These were categorized as "Most Risk". 

Because the variable "length" had many different values, it was also chosen to be more categorized so a new variable "employment_length" was created. The lengths "< 1 year", "1 year", "2 years", and "3 years" were categorized as the value "0 - 3 years" in the variable "employment_length". The lengths "4 years", "5 years", and "6 years" were categorized as the value "4 - 6 years". The lengths "7 years", "8 years", and "9 years" were categorized as the value "7 - 9 years". The length "10 +years" was categorized as the value "10+ years". Lastly, the length "NA" was categorized as the value "Not Applicable" in the variable "employment_length" since there can be reasons why a value could be missing, such as being unemployed.

With so many states listed out in the variable "state", it was best to categorize them into regions in the new variable "region". All the states that are in the Midwest region of the United States were categorized as "Midwest" in the "region" variable. All the states that are in the Northeast region of the United States were categorized as "Northeast". All the states that are in the South region of the United States were categorized as "South". All the states that are in the West region of the United States were categorized as "West".

The variable "reason" had many different levels so the categories with the smaller counts: car, house, moving, renewable_energy, small_business, vacation, wedding, and medical were all lumped into the "other" category. This will help us focus on the larger count catergories when using this variable as a predictor. 

```{r echo=FALSE}
new_loans <-
  loans %>%
  mutate(response = case_when(
    status == "Fully Paid" ~ "Good", 
    status == "Charged Off" | status =="Default" ~ "Bad",
    TRUE ~ NA_character_),
        risk = case_when(
          grade == "A" | grade == "B" ~ "Least Risk",
          grade == "C" | grade == "D" | grade == "E" ~ "Medium Risk",
          grade == "F" | grade == "G" ~ "Most Risk"),
        loan_length = case_when(
          length == "< 1 year" | length == "1 year" | length == "2 years" | length == "3 years" ~ "0 - 3 years",
          length == "4 years" | length == "5 years" | length == "6 years" ~ "4 - 6 years",
          length == "7 years" | length == "8 years" | length == "9 years" ~ "7 - 9 years",
          length == "10+ years" ~ "10+ years",
          length == NA ~ "Not Applicable"),
        region = case_when(
          state == "IA" | state == "IL" | state == "IN" | state == "KS" | state == "MI" | state == "MN" | state == "MO" | state == "ND" | state == "NE" | state == "OH" | state == "SD" | state == "WI" ~ "Midwest",
          state == "CT" | state == "MA" | state == "ME" | state == "NH" | state == "NJ" | state == "NY" | state == "PA" | state == "RI" | state == "VT" ~ "Northeast",
          state == "AL" | state == "AR" | state == "DC" | state == "DE" | state == "FL" | state == "GA" | state == "KY" | state == "LA" | state == "MD" | state == "MS" | state == "NC" | state == "OK" | state == "SC" | state == "TN" | state == "TX" | state == "VA" | state == "WV" ~ "South",
          state == "AK" | state == "AZ" | state == "CA" | state == "CO" | state == "HI" | state == "ID" | state == "MT" | state == "NM" | state == "NV" | state == "OR" | state == "UT" | state == "WA" | state == "WY" ~ "West")
         
        )
new_loans$reason[which(new_loans$reason == "car")] = "other"
new_loans$reason[which(new_loans$reason == "house")] = "other"
new_loans$reason[which(new_loans$reason == "moving")] = "other"
new_loans$reason[which(new_loans$reason == "renewable_energy")] = "other"
new_loans$reason[which(new_loans$reason == "small_business")] = "other"
new_loans$reason[which(new_loans$reason == "vacation")] = "other"
new_loans$reason[which(new_loans$reason == "wedding")] = "other"
new_loans$reason[which(new_loans$reason == "medical")] = "other"
```

The following variables will be removed from the dataset: employment, verified, bcOpen, bcRatio, totalBcLim, status, grade, length, and state. Employment will be removed because there are 21400 unique values within this variable, and without a clear way to categorize, will make using it as a predictor very difficult. Verified will be removed because whether a person's income can be verified is not good predictor. BcOpen, bcRatio, and totalBcLim will be removed because limiting to credit cards is too specific and the same information will be included in the other variables relating to credit. Status, grade, length, and state will be removed because new variables have been created to replace these variables. RevolRatio, totalAcc, totalRevLim, totalLim, totalRevBal, and totalIlLim will also be removed because there's alot of variables relating to credit, and credit information can mostly be covered by the variable "totalBal".


```{r echo=FALSE}
new_loans = new_loans[,!(names(new_loans) %in% c("employment", "verified", "bcOpen", "bcRatio", "totalBcLim", "status", "grade", "length", "state", "revolRatio", "totalAcc", "totalRevLim", "totalLim", "totalRevBal", "totalIlLim"))]
```

Loans that are late, current (being paid), or in grace period were displayed as NA's in the dataset. They have been deleted from the dataset. Also, there is about 1,823 values within the loan_length variable that are NA. Because the amount of missing values is pretty small compared to the size of the dataset, about 0.05, they will also be removed from the dataset.
```{r echo=FALSE}
new_loans <- new_loans[-which(is.na(new_loans$response)),]
sapply(new_loans, function(x) sum(is.na(x)))
new_loans <- new_loans[-which(is.na(new_loans$loan_length)),]
```

##Exploring and Transforming the data

To check for skewness, histograms were created for all numeric variables. Based off of the histograms, the rate variable looks slightly skewed to the right while the following variables look extrememly skewed to the right: delinq2yr, inq6mth, and pubRec.
```{r echo=FALSE}
new_loans %>%
  purrr::keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()
```


Because they all have extreme skewness to the right, I will use logarithms to transform the variables and reduce the skewness.

```{r}
new_loans$delinq2yr <- log(new_loans$delinq2yr + 1)
new_loans$inq6mth <- log(new_loans$inq6mth + 1)
new_loans$pubRec <- log(new_loans$pubRec + 1)
```

For the quantitative predictors, side-by-side histograms were created to compare the variables relationships between good and bad loans. There are not that many predictors that behaved differently for good and bad loans, but the variable "rate" does show more right skewness for good loans compared to bad loans.


```{r echo=FALSE}
par(mfrow=c(4,2),mar=c(3,3,2,1))
hist(new_loans$accOpen24[new_loans$response=="Good"], main = "Open Accounts vs. Good Loans")
hist(new_loans$accOpen24[new_loans$response=="Bad"], main = "Open Accounts vs. Bad Loans")
hist(new_loans$amount[new_loans$response=="Good"], main = "Loan Amount vs. Good Loans")
hist(new_loans$amount[new_loans$response=="Bad"], main = "Loan Amount vs. Bad Loans")
hist(new_loans$avgBal[new_loans$response=="Good"], main = "Average Balance vs. Good Loans")
hist(new_loans$avgBal[new_loans$response=="Bad"], main = "Average Balance vs. Bad Loans")
hist(new_loans$debtIncRat[new_loans$response=="Good"], main = "Debt Income Ratio vs. Good Loans")
hist(new_loans$debtIncRat[new_loans$response=="Bad"], main = "Debt Income Ratio vs. Bad Loans")
hist(new_loans$delinq2yr[new_loans$response=="Good"], main = "Number of Late Payments  vs. Good Loans")
hist(new_loans$delinq2yr[new_loans$response=="Bad"], main = "Number of Late Payments vs. Bad Loans")
hist(new_loans$income[new_loans$response=="Good"], main = "Income vs. Good Loans")
hist(new_loans$income[new_loans$response=="Bad"], main = "Income vs. Bad Loans")
hist(new_loans$inq6mth[new_loans$response=="Good"], main = "Number of Inquiries vs. Good Loans")
hist(new_loans$inq6mth[new_loans$response=="Bad"], main = "Number of Inquiries vs. Bad Loans")
hist(new_loans$openAcc[new_loans$response=="Good"], main = "Number of Credit Lines vs. Good Loans")
hist(new_loans$openAcc[new_loans$response=="Bad"], main = "Number of Credit Lines vs. Bad Loans")
hist(new_loans$payment[new_loans$response=="Good"], main = "Monthly Payment vs. Good Loans")
hist(new_loans$payment[new_loans$response=="Bad"], main = "Monthly Payment vs. Bad Loans")
hist(new_loans$pubRec[new_loans$response=="Good"], main = "Number of Derog.Public Rec vs. Good Loans")
hist(new_loans$pubRec[new_loans$response=="Bad"], main = "Number of Derog.Public Rec vs. Bad Loans")
hist(new_loans$rate[new_loans$response=="Good"], main = "Interest Rate vs. Good Loans")
hist(new_loans$rate[new_loans$response=="Bad"], main = "Interest Rate vs. Bad Loans")
hist(new_loans$totalBal[new_loans$response=="Good"], main = "Credit Balance vs. Good Loans")
hist(new_loans$totalBal[new_loans$response=="Bad"], main = "Credit Balance vs. Bad Loans")
hist(new_loans$totalBal[new_loans$response=="Good"], main = "Credit Balance vs. Good Loans")
hist(new_loans$totalBal[new_loans$response=="Bad"], main = "Credit Balance vs. Bad Loans")
```

For the catergorical predictors, bar graphs were created to show how the catergorical distribution varies for good and bad loans. Proportions were used instead of counts as the measurement to make the comparison between good loans and bad loans easier with the larger amount of good loans within the data. The predictors "term" and "risk" showed the most changed behaviors between all the categorical predictors. For the predictor "term", the proportion of 36 month terms increases from bad to good loans by 21.8%, while the proportion of 60 month terms decreases from bad to good loans by 21.8%. For the predictor "risk" the proportion of least risk increases from bad to good loans by 28.4%, medium risk decreases from bad to good loans by 22.9%, and high risk decreases from bad to good loans by 5.6%.


```{r echo=FALSE}
p1 <- ggplot(new_loans, aes(x= term,  group=response)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
        geom_text(aes( label = scales::percent(..prop..),
                       y= ..prop.. ), stat= "count", vjust = -.5, size = 3) +
        labs(y = "Percent", fill="term") +
        facet_grid(~response) +
        scale_y_continuous(labels = scales::percent) + theme(axis.text.x = element_text(angle = 90))
        
p2 <- ggplot(new_loans, aes(x= home,  group=response)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
        geom_text(aes( label = scales::percent(..prop..),
                       y= ..prop.. ), stat= "count", vjust = -.5, size = 3) +
        labs(y = "Percent", fill="home") +
        facet_grid(~response) +
        scale_y_continuous(labels = scales::percent) + theme(axis.text.x = element_text(angle = 90))
p3 <- ggplot(new_loans, aes(x= reason,  group=response)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
        geom_text(aes( label = scales::percent(..prop..),
                       y= ..prop.. ), stat= "count", vjust = -.5, size = 3) +
        labs(y = "Percent", fill="reason") +
        facet_grid(~response) +
        scale_y_continuous(labels = scales::percent) +
        scale_x_discrete(labels = abbreviate) + theme(axis.text.x = element_text(angle = 90))
p4 <- ggplot(new_loans, aes(x= risk,  group=response)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
        geom_text(aes( label = scales::percent(..prop..),
                       y= ..prop.. ), stat= "count", vjust = -.5, size = 3) +
        labs(y = "Percent", fill="risk") +
        facet_grid(~response) +
        scale_y_continuous(labels = scales::percent) + theme(axis.text.x = element_text(angle = 90))
p5 <- ggplot(new_loans, aes(x= loan_length,  group=response)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
        geom_text(aes( label = scales::percent(..prop..),
                       y= ..prop.. ), stat= "count", vjust = -.5, size = 3, angle = 90) +
        labs(y = "Percent", fill="loan_length") +
        facet_grid(~response) +
        scale_y_continuous(labels = scales::percent) + theme(axis.text.x = element_text(angle = 90))
p6 <- ggplot(new_loans, aes(x= region,  group=response)) + 
        geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") +
        geom_text(aes( label = scales::percent(..prop..),
                       y= ..prop.. ), stat= "count", vjust = -.5, size = 3, angle = 90) +
        labs(y = "Percent", fill="region") +
        facet_grid(~response) +
        scale_y_continuous(labels = scales::percent) + theme(axis.text.x = element_text(angle = 90))
grid.arrange(p1, p2,nrow = 1)
grid.arrange(p3, p4,nrow = 1)
grid.arrange(p5, p5,nrow = 1)
```

##The Logistic Model
Two datasets were created from the cleaned and prepared data from the previous steps. 80% of the cases were randomly chosen and used to create the dataset "train". The variable "totalPaid" was removed because since it can only be determined after a loan is issued, it can not be a predictor. The remaining 20% of the cases were used to create the dataset "test". A set.seed was also used to keep the results consistent each time the script is ran.

```{r}
set.seed(321) 
sample <- sample.int(n = nrow(new_loans), size = floor(.80*nrow(new_loans)), replace = F)
train <- new_loans[sample, ]
train = train[,!(names(train) %in% c("totalPaid"))]
test  <- new_loans[-sample, ]
```

Before the selected predictors can be used in a logistic regression, they have to be in the correct data type. The predictors can not have the data type of character, so all the character data types have been converted to factors.
A logistic regression model named "loan_fit" has been created using the dataset "train", that will use the selected predictors to predcit the loan status. The model was then used to predict the loan status. The probability of each of loan being good was created and stored in the data set "loan_prob". If a value in "loan_prob" was more than .5, a value of "Good" was added to the data set "loan_pred" and if a value was .5 or less, a value of "Bad" was added to "loan_pred". 

```{r}
train[sapply(train, is.character)] <- lapply(train[sapply(train, is.character)], as.factor)
loan_fit <- glm(response~., data = train, family = "binomial")
loan_prob <- predict(loan_fit, test, type = "response")
loan_pred <- ifelse(loan_prob > 0.5, "Good", "Bad")
```

A contingency table was created to help visualize the accuracy of the data. The overall accuracy of the model was calculated and the results are that the model is about 79% accurate. About 12% of the bad loans were correctly predicted as bad and about 97.4% of the good loans were correctly predicted as good.

```{r echo=FALSE}
tbl <- table(loan_pred, test$response)
addmargins(tbl)
mean(loan_pred == test$response)
```

##Optimizing the Threshold Accuracy

A function was created to see how the overall accuracy will change depending on the threshold. The function calculated the accuracies starting from the threshold of .1 to the threshold of .9 and incremented by .05. A plot was created to see how the accuracy changed by the threshold. Seen by the plot, the accuracy reaches it's maximum point at .5, but after .5 the accuracy seems to decease drastically as the threshold increases. I believe a threshold of .5 will produce the maximum accuracy at 79.37%.

```{r echo=FALSE}
threshold <- seq(.1,.9,.05)
accuracy <- function(thresh){
  mean(ifelse(loan_prob > thresh, "Good", "Bad") == test$response)
}
accuracy.prediction <- sapply(threshold, accuracy)
plot(threshold, accuracy.prediction, main = "Accuracy Vs. Threshold", xlab = "Threshold", ylab = "Accuracy")
accuracy(.5)
```

##Optimizing the Threshold for Profit

A test was ran to see how the bank's profit will change depending on the threshold. First a new variable was added to the test data set, "pred.prob" that will show the probability of the loan to be good. A function was created where according to the threshold selected, will add the value 1 to the data set "pred.outcome" if the probability was more than the threshold and 0 if it wasn't. The function will sum the "totalPaid" of the the 1's and subtract the sum of the "amount" of all the 1's from it. This will calculate the profit. The function will go through all the thresholds from .1 to .9 with increments of .05 to show how the profit will change based off of the specified threshold. Seen by the plot, from the threshold .1 to .7 the profit increases, but for all the thresholds after .7, the profit decreases. I believe a threshold of .7 will produce the most profit at $3,870,539. 

```{r echo=FALSE}
test$pred.prob <- predict(loan_fit, test, type = "response")
profit <- function(thresh){
  pred.outcome = ifelse(test$pred.prob > thresh, 1, 0)
sum(test$totalPaid[which(pred.outcome == 1)]) - sum(test$amount[which(pred.outcome == 1)])
}
profit.prediction <- sapply(threshold, profit)
par(mar=c(4,10,2,2))
options(scipen=100)
plot(threshold, profit.prediction, main = "Profit Vs. Threshold", xlab = "Threshold", ylab = "", las=1)
mtext(text = "Profit",
      side = 2,
      line = 5)
profit(.7)
```

This maximum profit threshold at .7 does not coincide with the maximum accuracy threshold at .5. At a threshold of .7, the accuracy is about 74.74%, compared to .5 with an accuracy of 79.37%.

```{r}
accuracy(.7)
```

The maximum possible profit is $60,123,881 
and the profit under the current process is $3,913,761. Compared to not using the model, the maximum percentage increase in profit that can be expected by deploying your model will be about 6.4%.

```{r}
max_profit <- (sum(new_loans$totalPaid[which(new_loans$response == "Good")]) - sum(new_loans$amount[which(new_loans$response == "Good")]))
max_profit
 profit(.7) / (sum(new_loans$totalPaid[which(new_loans$response == "Good")]) - sum(new_loans$amount[which(new_loans$response == "Good")]))
```

##Results Summary

The purpose of this study is to predict which applicants are likely to default on their loans. Using the variables: amount, term, rate, payment, home, income, reason, debyIncRat, delinq2yr, inq6mth, openAcc, pubRec, totalPaid, totalBal, accOpen24, avgBal, response, risk, loan_length and region from the dataset loans, each loan was predicted to be good or bad. The logistic regression was first created and then the threshold for the prediction was tested to see which one yielded the highest accuracy and profit.The final model will include the threshold, .5, that yields the highest overall accuracy, 79.23%. This will yield a profit of $2,410,333. The model is able to correctly predict bad loans 9.4% and correctly predict good loans 98%. 
There are limitations to this model. Since the highest accurate rate and the maximum profit from the model does not share the same threshold, our model will never be able to produce the highest accurate and maxiumum profit at the same time. Also any incorrect computations done previously will cause innacuracies produced from the model.
