#Loading required libraries
library(mice)
library(gains)
library(dplyr)
library(ROCR)
library(caret)

#Reading data from sample csv file
creditdata<-read.csv("C:\\Users\\Ronit S. Mhatre\\Desktop\\R\\logistic regression\\sample.csv",header=T,na.strings=c(""),stringsAsFactors=FALSE)

#Checking the datatypes
str(creditdata)

#Checking the data summary
summary(creditdata)

#Converting to appropriate datatypes
creditdata$MonthlyIncome <- as.numeric(creditdata$MonthlyIncome)
creditdata$MonthlyIncome.1 <- as.numeric(creditdata$MonthlyIncome.1)
creditdata$NumberOfDependents <- as.integer(creditdata$NumberOfDependents)
creditdata$Rented_OwnHouse <- as.factor(creditdata$Rented_OwnHouse)
creditdata$Occupation <- as.factor(creditdata$Occupation)
creditdata$Education <- as.factor(creditdata$Education)
creditdata$Gender <- as.factor(creditdata$Gender)
creditdata$Region <- as.factor(creditdata$Region)


#Analysing missing values
md.pattern(creditdata)

#Total records - 2000
#MonthlyIncome missing -396
#No of dependents missing -62

#Imputing the missing data
tempdata <- mice(creditdata,m=5,maxit=50,meth='pmm',seed=500)
summary(tempdata)
completeddata <- complete(tempdata,1)
md.pattern(completeddata)


#Add Age and Monthly income in ranges
completeddata$AgeRange <- as.factor(ifelse(completeddata$age>20 & completeddata$age<=40,'21-40',
                                           ifelse(completeddata$age<=60,'41-60',
                                                  ifelse(completeddata$age<=80,'61-80','81-100'))))
completeddata$MonthlyIncomeRange <- as.factor(ifelse(completeddata$MonthlyIncome<2500,'<2500',
                                       ifelse(completeddata$MonthlyIncome<=5000,'2500-5000',
                                              ifelse(completeddata$MonthlyIncome<=7500,'5000-7500',
                                                     ifelse(completeddata$MonthlyIncome<=10000,'7500-10000','>10000')))))


#Splitting into test and training samples
set.seed(200)
index<-sample(nrow(completeddata),0.70*nrow(completeddata),replace=F)
train<-completeddata[index,]
test<-completeddata[-index,]



#Applying Logistic Regression initially for all variables
logmod<-glm(formula=NPA.Status~RevolvingUtilizationOfUnsecuredLines + AgeRange + 
              Gender + Region + MonthlyIncomeRange + Rented_OwnHouse + Occupation + 
              Education + NumberOfTime30.59DaysPastDueNotWorse + DebtRatio + 
              NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate + 
              NumberRealEstateLoansOrLines + NumberOfTime60.89DaysPastDueNotWorse + 
              NumberOfDependents, data=train, family="binomial")

#Analysing the result
summary(logmod)

#Check the variables using stepwise procedure
step(logmod,direction="both")


#Model with the suggested variables from stepwise procedure
logmod1<-glm(formula = NPA.Status ~ RevolvingUtilizationOfUnsecuredLines + 
                AgeRange + Region + MonthlyIncomeRange + Rented_OwnHouse + 
                Education + NumberOfTime30.59DaysPastDueNotWorse + NumberOfTimes90DaysLate + 
                NumberOfTime60.89DaysPastDueNotWorse, family = "binomial", 
                data = train)

#Analysing the result
summary(logmod1)

#Adding Dummy Variables
train$AgeRange61to80 <- ifelse(train$AgeRange=='61-80',1,0)
train$Education_Graduate <- ifelse(train$Education=='Graduate',1,0)
train$Education_Matric <- ifelse(train$Education=='Matric',1,0)
train$Education_PhD <- ifelse(train$Education=='PhD',1,0)
train$Region_Central <- ifelse(train$Region=='Central',1,0)
train$Region_West <- ifelse(train$Region=='West',1,0)
train$Region_East <- ifelse(train$Region=='East',1,0)
train$Region_North <- ifelse(train$Region=='North',1,0)

test$AgeRange61to80 <- ifelse(test$AgeRange=='61-80',1,0)
test$Education_Graduate <- ifelse(test$Education=='Graduate',1,0)
test$Education_Matric <- ifelse(test$Education=='Matric',1,0)
test$Education_PhD <- ifelse(test$Education=='PhD',1,0)
test$Region_Central <- ifelse(test$Region=='Central',1,0)
test$Region_West <- ifelse(test$Region=='West',1,0)
test$Region_East <- ifelse(test$Region=='East',1,0)
test$Region_North <- ifelse(test$Region=='North',1,0)

#Build model with dummy variables
logmod2<-glm(formula = NPA.Status ~ AgeRange61to80 + Region_Central + Region_West  + Region_North + Rented_OwnHouse + 
                Education_Matric + Education_PhD +  
               NumberOfTime30.59DaysPastDueNotWorse + NumberOfTimes90DaysLate + 
               NumberOfTime60.89DaysPastDueNotWorse, family = "binomial", 
             data = train)

#Analysing the result
summary(logmod2)

step(logmod2,direction="both")

pred<-predict(logmod2,type="response",newdata=test)

head(pred)

#Find the percentage of NPA.Status = 1
table(completeddata$NPA.Status)/nrow(completeddata)
pred<-ifelse(pred>=0.0715,1,0)

#Cross Join of predicted events vs the NPA.Status
table(pred,test$NPA.Status)

#Concordance function
Concordance = function(GLM.binomial) {
  outcome_and_fitted_col = cbind(GLM.binomial$y, GLM.binomial$fitted.values)
  # get a subset of outcomes where the event actually happened
  ones = outcome_and_fitted_col[outcome_and_fitted_col[,1] == 1,]
  # get a subset of outcomes where the event didn't actually happen
  zeros = outcome_and_fitted_col[outcome_and_fitted_col[,1] == 0,]
  # Equate the length of the event and non-event tables
  if (length(ones[,1])>length(zeros[,1])) {ones = ones[1:length(zeros[,1]),]}
  else {zeros = zeros[1:length(ones[,1]),]}
  # Following will be c(ones_outcome, ones_fitted, zeros_outcome, zeros_fitted)
  ones_and_zeros = data.frame(ones, zeros)
  # initiate columns to store concordant, discordant, and tie pair evaluations
  conc = rep(NA, length(ones_and_zeros[,1]))
  disc = rep(NA, length(ones_and_zeros[,1]))
  ties = rep(NA, length(ones_and_zeros[,1]))
  for (i in 1:length(ones_and_zeros[,1])) {
    # This tests for concordance
    if (ones_and_zeros[i,2] > ones_and_zeros[i,4])
    {conc[i] = 1
    disc[i] = 0
    ties[i] = 0}
    # This tests for a tie
    else if (ones_and_zeros[i,2] == ones_and_zeros[i,4])
    {
      conc[i] = 0
      disc[i] = 0
      ties[i] = 1
    }
    # This should catch discordant pairs.
    else if (ones_and_zeros[i,2] < ones_and_zeros[i,4])
    {
      conc[i] = 0
      disc[i] = 1
      ties[i] = 0
    }
  }
  # Here we save the various rates
  conc_rate = mean(conc, na.rm=TRUE)
  disc_rate = mean(disc, na.rm=TRUE)
  tie_rate = mean(ties, na.rm=TRUE)
  return(list(concordance=conc_rate, num_concordant=sum(conc), discordance=disc_rate, num_discordant=sum(disc), tie_rate=tie_rate,num_tied=sum(ties)))
  
}


#Analysing Concordance - 0.7981651
Concordance(logmod2)

#Analysing gains Chart - 30% of data predicts 76.5% of the events
gains(test$NPA.Status,predict(logmod2,type="response",newdata=test),groups = 10)

#AUC
Pred<-prediction(predict(logmod2,newdata = test,type = "response"),test$NPA.Status)
perf<-performance(Pred,"auc")
unlist(perf@y.values)
