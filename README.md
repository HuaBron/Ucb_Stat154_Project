# Ucb_Stat154_Project2
This is an interesting project about clouds exploration in Arctic finished by **Yang Xiang** and **Zhihua Zhang**. During finishing this project, we refer to many resources such as R document [_trainModelList_](http://topepo.github.io/caret/train-models-by-tag.html#Support_Vector_Machines) and some great r blogs like [_Evaluating Logistic Regression Models_](https://www.r-bloggers.com/evaluating-logistic-regression-models/) for help.

To proceed our project, the whole work is divided into two main parts:
1. Preparation: 
 * First we read the reference paper carefully, exchange views and try to have a better domain sketch.
 * Because the data is cleaned, we then begin to do visualization and EDA. By simply plotting the well-labeled map according to expert labels(including the uncertain data), we observe the existence of **spatial dependency**.
 * Considering the existence of **non-i.i.d** data, instead of splitting data randomly, we do **block-sampling** to divide. To be specific, it includes partitioning data into blocks to remain dependency and then doing random sampling. The size of block is the key parameter to consider and we choose by adjusting and comparing many times.
 * After that, we check the performance of trivial classifier by assigning all the validation and test data to label _-1_, and find out the problem at hand is **non trivial**.
 * As good features are cricual to obtain higher accuracy, we select them through quantitative and visual justification like calculating the correlation and making plot based on every features. Finally we decide to use three best **_NDAI_**, **_SD_** and **_CORR_**.
 * CV is indeed a terrific way to evaluate our model, and a customized CV function could do us even more favor. Thus, we design our own **CV generic** which can fit many models like lda, qda, logistic regression and svm, and reports final and individual fold accuracy.

2. Model training and result evaluation:
