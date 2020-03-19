# Ucb_Stat154_Project2
This is an interesting project about clouds exploration in Arctic finished by **Yang Xiang** and **Zhihua Zhang**. During finishing this project, we refer to many resources such as R document [_trainModelList_](http://topepo.github.io/caret/train-models-by-tag.html#Support_Vector_Machines), some great r blogs like [_Evaluating Logistic Regression Models_](https://www.r-bloggers.com/evaluating-logistic-regression-models/) and [_SVM Wikipeida_](https://en.wikipedia.org/wiki/Support-vector_machine) for help.

To proceed our project, the whole work is divided into two main parts:
1. Preparation: 
 * R code in part 1/(b) verifies the certain(available) data we have and makes well-labeled map based on expert labels to ascertain whether there exists spatial dependency.
 * R code in part 1/(c) does EDA using "ggpair" to check pairwise relationship among different features and expert labels.
 * R code in part 2/(a) aims at spliting non i.i.d cloud data, which is the block-sampling technique introduced in the report. And we try two division with different block sizes, 225 and 625.
 * R code in part 2/(b) inspects whether the problem at hand is trivial by assigning all validation and test data to label "-1" and evaluating the accuracy.
 * R code in part 2/(c) proposes the three best features used in the later model training by implementing lda(the result is same if trying Logistic regression and SVM) and comparing the variable importance.
 * R code in part 2/(d) designs our customised CV generic function to fit different models(Logistic regerssion, lda, qda, SVM, etc) and return final and individual accuracies.

2. Model training and prediction
 * R code Part 3 Modeling includes models we have tried, including LDA, QDA, Logistic Regression, Linear SVM and Kernel SVM.
 * After training the model and doing prediction, running code in ROC curve helps generate the plots of ROC curves of all models.
 * Code in Assessing models by confusion matrix helps generate confusion matrixes and F1 score of all models.

3. Diagnostic analysis
 * R code in Part 4/(a) generates the plot of parameters and accuracy convergence of Kernel SVM.
 * R code in part 4/(b) generates the plot of patterns of misclassified data, including two histograms of NDAI and a coordinate plot.
 * R code in part 4/(c) generates the result of mixture model (QDA and Kernel SVM with weights 0.5 and 0.5).
 * By changing parameters in previous part of data split from 225 to 625, we can get results of 4/(d), whether the results change if we change the way of splitting data.
