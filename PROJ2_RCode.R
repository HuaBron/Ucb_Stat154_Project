library(ggplot2)
library(ggpubr)
library(GGally)
library(gridExtra)
library(MASS)
library(e1071)
library(caret)
library(kernlab)
library(plotROC)
library(pROC)
library(MVN)


####################################################
# Preparation Part
####################################################


# Read original data
Image1 = read.table(file = '/Users/heart/Desktop/learning/ML&Prediction/PROJ2_write-up/image1.txt',header = F)
Image2 = read.table(file = '/Users/heart/Desktop/learning/ML&Prediction/PROJ2_write-up/image2.txt',header = F)
Image3 = read.table(file = '/Users/heart/Desktop/learning/ML&Prediction/PROJ2_write-up/image3.txt',header = F)
# AllImage = rbind(Image1,Image2,Image3)
# colnames(AllImage) <- c('x','y','ex_label','NDAI','SD','CORR',"Ra_{DF}",'Ra_CF',"Ra_{BF}",'Ra_AF','Ra_AN')
# AllImage$ex_label = factor(AllImage$ex_label)

colnames(Image1) <- c('x','y','ex_label','NDAI','SD','CORR',"Ra_DF",'Ra_CF',"Ra_BF",'Ra_AF','Ra_AN')
Image1$ex_label = factor(Image1$ex_label)
colnames(Image2) <- c('x','y','ex_label','NDAI','SD','CORR',"Ra_DF",'Ra_CF',"Ra_BF",'Ra_AF','Ra_AN')
Image2$ex_label = factor(Image2$ex_label)
colnames(Image3) <- c('x','y','ex_label','NDAI','SD','CORR',"Ra_DF",'Ra_CF',"Ra_BF",'Ra_AF','Ra_AN')
Image3$ex_label = factor(Image3$ex_label)


# The percent of data labeled by experts
ClassPercent = matrix(0,3,3)
ClassPercent[1,] = c(sum(Image1$ex_label==-1),sum(Image1$ex_label==0),sum(Image1$ex_label==1))/dim(Image1)[1]
ClassPercent[2,] = c(sum(Image2$ex_label==-1),sum(Image2$ex_label==0),sum(Image2$ex_label==1))/dim(Image2)[1]
ClassPercent[3,] = c(sum(Image3$ex_label==-1),sum(Image3$ex_label==0),sum(Image3$ex_label==1))/dim(Image3)[1]
ClassPercent


# Well-labeled map based on Expert Labels
Image1$ex_label = factor(Image1$ex_label)
p1 = ggplot(data = Image1) + geom_point(aes(x=x,y=y,col=ex_label),show.legend = T) + xlab('x coordinate') + ylab('y coordinate') + ggtitle('Expert labels visualization of Image 1') + theme(plot.title = element_text(hjust = 0.5, size = 35))+labs()
Image2$ex_label = factor(Image2$ex_label)
p2 = ggplot(data = Image2) + geom_point(aes(x=x,y=y,col=ex_label),show.legend = T) + xlab('x coordinate') + ylab('y coordinate') + ggtitle('Expert labels visualization of Image 2') + theme(plot.title = element_text(hjust = 0.5, size = 35))+labs()
Image3$ex_label = factor(Image3$ex_label)
p3 = ggplot(data = Image3) + geom_point(aes(x=x,y=y,col=ex_label),show.legend = T) + xlab('x coordinate') + ylab('y coordinate') + ggtitle('Expert labels visualization of Image 3') + theme(plot.title = element_text(hjust = 0.5, size = 35))+labs()
grid.arrange(p1,p2,p3,nrow=2)


# Pairwise Relationship Plot
pp1 = ggpairs(Image1,columns = 3:11,title='Pairwise relationship of Image 1',ggplot2::aes(colour=ex_label)) + theme(plot.title = element_text(hjust = 0.5, size = 35))
pp1
pp2 = ggpairs(Image2,columns = 3:11,title='Pairwise relationship of Image 2',ggplot2::aes(colour=ex_label)) + theme(plot.title = element_text(hjust = 0.5, size = 35))
pp2
pp3 = ggpairs(Image3,columns = 3:11,title='Pairwise relationship of Image 3',ggplot2::aes(colour=ex_label)) + theme(plot.title = element_text(hjust = 0.5, size = 35))
pp3


# Extract the data with expert labels
ExImage1 = Image1[which(Image1$ex_label!=0),]
ExImage1$ex_label = factor(ExImage1$ex_label)
ExImage2 = Image2[which(Image2$ex_label!=0),]
ExImage2$ex_label = factor(ExImage2$ex_label)
ExImage3 = Image3[which(Image3$ex_label!=0),]
ExImage3$ex_label = factor(ExImage3$ex_label)


###### General function to split CloudData
DataSplit = function(CloudData,BlockNum){
  set.seed(2019)
  RowNum = ColNum = sqrt(BlockNum)
  xQuantile = quantile(x = CloudData$x,probs = seq(0,1,length.out = ColNum))
  yQuantile = quantile(x = CloudData$y,probs = seq(0,1,length.out = RowNum))
  BlockList = list()
  for(i in 1:RowNum){
    for(j in 1:ColNum){
      ### Include last point to avoid losing info
      xRange = c(xQuantile[i],xQuantile[i+1]-sum(i<RowNum))
      yRange = c(yQuantile[j],yQuantile[j+1]-sum(j<ColNum))
      PixelChoice = intersect(which(xRange[1]<=CloudData$x & CloudData$x <=xRange[2]),which(yRange[1]<=CloudData$y & CloudData$y <=yRange[2]))
      BlockList[[(i-1)*ColNum+j]] = CloudData[PixelChoice,]
    }
  }
  ### Split to three sets: 6/2/2
  TrainSize = ceiling(BlockNum*0.6)
  ValSize = floor(BlockNum*0.2)
  Shuffle = sample(1:BlockNum)
  TrainData = BlockList[[Shuffle[1]]]
  for(i in Shuffle[2:TrainSize]){
    TrainData = rbind(TrainData,BlockList[[i]])
  }
  ValData = BlockList[[Shuffle[TrainSize+1]]]
  for(i in Shuffle[(TrainSize+2):(TrainSize+ValSize)]){
    ValData = rbind(ValData,BlockList[[i]])
  }
  TestData = BlockList[[Shuffle[TrainSize+ValSize+1]]]
  for(i in Shuffle[(TrainSize+ValSize+2):BlockNum]){
    TestData = rbind(TestData,BlockList[[i]])
  }
  
  SplitDataList = list(TrainData,ValData,TestData)
  names(SplitDataList) = c('train','val','test')
  return(SplitDataList)
}


# Use 225 blocks to split
BlockNum1 = 15^2
Image1.sp1 = DataSplit(ExImage1,BlockNum1)
Image2.sp1 = DataSplit(ExImage2,BlockNum1)
Image3.sp1 = DataSplit(ExImage3,BlockNum1)


# Use 625 blocks, smaller one, to split
BlockNum2 = 25^2
Image1.sp2 = DataSplit(ExImage1,BlockNum2)
Image2.sp2 = DataSplit(ExImage2,BlockNum2)
Image3.sp2 = DataSplit(ExImage3,BlockNum2)


# Trivial classifier trained in data divided by method 1
###### Image 1
ValTriv1.sp1 = sum(Image1.sp1$val$ex_label==-1)/dim(Image1.sp1$val)[1]
TestTriv1.sp1 = sum(Image1.sp1$test$ex_label==-1)/dim(Image1.sp1$test)[1]

###### Image 2
ValTriv2.sp1 = sum(Image2.sp1$val$ex_label==-1)/dim(Image2.sp1$val)[1]
TestTriv2.sp1 = sum(Image2.sp1$test$ex_label==-1)/dim(Image2.sp1$test)[1]

###### Image 3
ValTriv3.sp1 = sum(Image3.sp1$val$ex_label==-1)/dim(Image3.sp1$val)[1]
TestTriv3.sp1 = sum(Image3.sp1$test$ex_label==-1)/dim(Image3.sp1$test)[1]

print(c(ValTriv1.sp1,TestTriv1.sp1))
print(c(ValTriv2.sp1,TestTriv2.sp1))
print(c(ValTriv3.sp1,TestTriv3.sp1))


# Trivial classifier trained in data divided by method 2
###### Image 1
ValTriv1.sp2 = sum(Image1.sp2$val$ex_label==-1)/dim(Image1.sp2$val)[1]
TestTriv1.sp2 = sum(Image1.sp2$test$ex_label==-1)/dim(Image1.sp2$test)[1]

###### Image 2
ValTriv2.sp2 = sum(Image2.sp2$val$ex_label==-1)/dim(Image2.sp2$val)[1]
TestTriv2.sp2 = sum(Image2.sp2$test$ex_label==-1)/dim(Image2.sp2$test)[1]

###### Image 3
ValTriv3.sp2 = sum(Image3.sp2$val$ex_label==-1)/dim(Image3.sp2$val)[1]
TestTriv3.sp2 = sum(Image3.sp2$test$ex_label==-1)/dim(Image3.sp2$test)[1]

print(c(ValTriv1.sp2,TestTriv1.sp2))
print(c(ValTriv2.sp2,TestTriv2.sp2))
print(c(ValTriv3.sp2,TestTriv3.sp2))


# Select three best features through lda
# Train in data divided by method 1
##### Image 1
Image1fit1.sp1 = train(ex_label~NDAI+SD+CORR+Ra_DF+Ra_CF+Ra_BF+Ra_AF+Ra_AN, data = Image1.sp1$train,method = 'lda',family = 'binomial')
Image1VI.sp1 = varImp(Image1fit1.sp1)
df1.lda = Image1VI.sp1$importance
df1.lda$All_Features = rownames(df1.lda)
p1 = ggplot(df1.lda)+geom_bar(aes(x=reorder(All_Features,-X1),y=X1),stat = 'identity')+ggtitle('Image1: Feature Importance in lda') + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(size = 9.8,angle = 45,vjust = 0.9,hjust = 0.9)) + xlab('')

Image1fit2.sp1 = train(ex_label~NDAI+SD+Ra_BF, data = Image1.sp1$train,method = 'lda',family = 'binomial')
Image1Pre = predict(Image1fit2.sp1,Image1.sp1$val)
Image1_Results = factor(Image1Pre==Image1.sp1$val$ex_label)
p2 = ggplot(Image1.sp1$val)+geom_point(aes(x=x,y=y,color=Image1_Results))+ggtitle('Classification by NDAI,SD,Ra_BF') + theme(plot.title = element_text(hjust = 0.5))

##### Image 2
Image2fit1.sp1 = train(ex_label~NDAI+SD+CORR+Ra_BF+Ra_CF+Ra_BF+Ra_AF+Ra_AN, data = Image2.sp1$train,method = 'lda',family = 'binomial')
Image2VI.sp1 = varImp(Image2fit1.sp1)
df2.lda = Image2VI.sp1$importance
df2.lda$All_Features = rownames(df2.lda)
p3 = ggplot(df2.lda)+geom_bar(aes(x=reorder(All_Features,-X1),y=X1),stat = 'identity')+ggtitle('Image2: Feature Importance in lda') + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(size = 9.8,angle = 45,vjust = 0.9,hjust = 0.9)) + xlab('')

Image2fit2.sp1 = train(ex_label~CORR+NDAI+SD, data = Image2.sp1$train,method = 'lda',family = 'binomial')
Image2Pre = predict(Image2fit2.sp1,Image2.sp1$val)
Image2_Results = factor(Image2Pre==Image2.sp1$val$ex_label)
p4 = ggplot(Image2.sp1$val)+geom_point(aes(x=x,y=y,color=Image2_Results))+ggtitle('Classification by CORR,NDAI,SD') + theme(plot.title = element_text(hjust = 0.5))

##### Image 3
Image3fit1.sp1 = train(ex_label~NDAI+SD+CORR+Ra_DF+Ra_CF+Ra_BF+Ra_AF+Ra_AN, data = Image3.sp1$train,method = 'lda',family = 'binomial')
Image3VI.sp1 = varImp(Image3fit1.sp1)
df3.lda = Image3VI.sp1$importance
df3.lda$All_Features = rownames(df3.lda)
p5 = ggplot(df3.lda)+geom_bar(aes(x=reorder(All_Features,-X1),y=X1),stat = 'identity')+ggtitle('Image3: Feature Importance in lda') + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(size = 9.8,angle = 45,vjust = 0.9,hjust = 0.9)) + xlab('')

Image3fit2.sp1 = train(ex_label~NDAI+SD+CORR, data = Image3.sp1$train,method = 'lda',family = 'binomial')
Image3Pre = predict(Image3fit2.sp1,Image3.sp1$val)
Image3_Results = factor(Image3Pre==Image3.sp1$val$ex_label)
p6 = ggplot(Image3.sp1$val)+geom_point(aes(x=x,y=y,color=Image3_Results))+ggtitle('Classification by NDAI,SD,CORR') + theme(plot.title = element_text(hjust = 0.5))

grid.arrange(p1,p2,p3,p4,p5,p6,nrow=3)


# Train in data divided by method 2
##### Image 1
Image1fit1.sp2 = train(ex_label~NDAI+SD+CORR+Ra_DF+Ra_CF+Ra_BF+Ra_AF+Ra_AN, data = Image1.sp2$train,method = 'lda',family = 'binomial')
Image1VI.sp2 = varImp(Image1fit1.sp2)
df1.lda = Image1VI.sp2$importance
df1.lda$All_Features = rownames(df1.lda)
p1 = ggplot(df1.lda)+geom_bar(aes(x=reorder(All_Features,-X1),y=X1),stat = 'identity')+ggtitle('Image1: Feature Importance in lda') + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(size = 9.8,angle = 45,vjust = 0.9,hjust = 0.9)) + xlab('')

Image1fit2.sp2 = train(ex_label~NDAI+SD+Ra_CF, data = Image1.sp2$train,method = 'lda',family = 'binomial')
Image1Pre = predict(Image1fit2.sp2,Image1.sp2$val)
Image1_Results = factor(Image1Pre==Image1.sp2$val$ex_label)
p2 = ggplot(Image1.sp2$val)+geom_point(aes(x=x,y=y,color=Image1_Results))+ggtitle('Classification by NDAI,SD,Ra_CF') + theme(plot.title = element_text(hjust = 0.5))

##### Image 2
Image2fit1.sp2 = train(ex_label~NDAI+SD+CORR+Ra_DF+Ra_CF+Ra_BF+Ra_AF+Ra_AN, data = Image2.sp2$train,method = 'lda',family = 'binomial')
Image2VI.sp2 = varImp(Image2fit1.sp2)
df2.lda = Image2VI.sp2$importance
df2.lda$All_Features = rownames(df2.lda)
p3 = ggplot(df2.lda)+geom_bar(aes(x=reorder(All_Features,-X1),y=X1),stat = 'identity')+ggtitle('Image2: Feature Importance in lda') + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(size = 9.8,angle = 45,vjust = 0.9,hjust = 0.9)) + xlab('')

Image2fit2.sp2 = train(ex_label~CORR+NDAI+SD, data = Image2.sp2$train,method = 'lda',family = 'binomial')
Image2Pre = predict(Image2fit2.sp2,Image2.sp2$val)
Image2_Results = factor(Image2Pre==Image2.sp2$val$ex_label)
p4 = ggplot(Image2.sp2$val)+geom_point(aes(x=x,y=y,color=Image2_Results))+ggtitle('Classification by CORR,NDAI,SD') + theme(plot.title = element_text(hjust = 0.5))

##### Image 3
Image3fit1.sp2 = train(ex_label~NDAI+SD+CORR+Ra_DF+Ra_CF+Ra_BF+Ra_AF+Ra_AN, data = Image3.sp2$train,method = 'lda',family = 'binomial')
Image3VI.sp2 = varImp(Image3fit1.sp2)
df3.lda = Image3VI.sp2$importance
df3.lda$All_Features = rownames(df3.lda)
p5 = ggplot(df3.lda)+geom_bar(aes(x=reorder(All_Features,-X1),y=X1),stat = 'identity')+ggtitle('Image3: Feature Importance in lda') + theme(plot.title = element_text(hjust = 0.5),axis.text.x = element_text(size = 9.8,angle = 45,vjust = 0.9,hjust = 0.9)) + xlab('')

Image3fit2.sp2 = train(ex_label~NDAI+SD+CORR, data = Image3.sp2$train,method = 'lda',family = 'binomial')
Image3Pre = predict(Image3fit2.sp2,Image3.sp2$val)
Image3_Results = factor(Image3Pre==Image3.sp2$val$ex_label)
p6 = ggplot(Image3.sp2$val)+geom_point(aes(x=x,y=y,color=Image3_Results))+ggtitle('Classification by NDAI,SD,CORR') + theme(plot.title = element_text(hjust = 0.5))

grid.arrange(p1,p2,p3,p4,p5,p6,nrow=3)



####################################################
# Model Training Part
####################################################

# Read data
setwd('/Users/xiangyang/Desktop/BerkeleyClasses/StatisticalPredictionAndMachineLearning/Proj/project2/image_data')
dat1 <- read.delim('image1.txt', header = FALSE, sep = "")
dat2 <- read.delim('image2.txt', header = FALSE, sep = "")
dat3 <- read.delim('image3.txt', header = FALSE, sep = "")
names(dat1) <- c('y','x','label','NDAI','SD','CORR',
                'DF','CF','BF','AF','AN')
names(dat2) <- c('y','x','label','NDAI','SD','CORR',
                'DF','CF','BF','AF','AN')
names(dat3) <- c('y','x','label','NDAI','SD','CORR',
                'DF','CF','BF','AF','AN')
dat <- rbind(dat1,dat2,dat3)
dat <- dat[dat$label != 0,]
dat$label <- factor(dat$label)

# EDA
p1 <- ggplot(dat1) + geom_point(aes(x = x,y = y, col = label))
p2 <- ggplot(dat2) + geom_point(aes(x = x,y = y, col = label))
p3 <- ggplot(dat3) + geom_point(aes(x = x,y = y, col = label))
ggarrange(p1,p2,p3,ncol = 2,nrow = 2)
ggpairs()

# Split data
DataSplit = function(CloudData,BlockNum){
  set.seed(2019)
  RowNum = ColNum = sqrt(BlockNum)
  xQuantile = quantile(x = CloudData$x,probs = seq(0,1,length.out = ColNum))
  yQuantile = quantile(x = CloudData$y,probs = seq(0,1,length.out = RowNum))
  BlockList = list()
  for(i in 1:RowNum){
    for(j in 1:ColNum){
      ### Include last point to avoid losing info
      xRange = c(xQuantile[i],xQuantile[i+1]-sum(i<RowNum))
      yRange = c(yQuantile[j],yQuantile[j+1]-sum(j<ColNum))
      PixelChoice = intersect(which(xRange[1]<=CloudData$x & CloudData$x <=xRange[2]),which(yRange[1]<=CloudData$y & CloudData$y <=yRange[2]))
      BlockList[[(i-1)*ColNum+j]] = CloudData[PixelChoice,]
    }
  }
  ### Split to three sets: 6/2/2
  TrainSize = ceiling(BlockNum*0.6)
  ValSize = floor(BlockNum*0.2)
  Shuffle = sample(1:BlockNum)
  TrainData = BlockList[[Shuffle[1]]]
  for(i in Shuffle[2:TrainSize]){
    TrainData = rbind(TrainData,BlockList[[i]])
  }
  ValData = BlockList[[Shuffle[TrainSize+1]]]
  for(i in Shuffle[(TrainSize+2):(TrainSize+ValSize)]){
    ValData = rbind(ValData,BlockList[[i]])
  }
  TestData = BlockList[[Shuffle[TrainSize+ValSize+1]]]
  for(i in Shuffle[(TrainSize+ValSize+2):BlockNum]){
    TestData = rbind(TestData,BlockList[[i]])
  }
  
  SplitDataList = list(TrainData,ValData,TestData)
  names(SplitDataList) = c('train','val','test')
  return(SplitDataList)
}

# Split Method 1
SplitDat = DataSplit(dat,225)
datTrain = SplitDat$train
datVal = SplitDat$val
datTest = SplitDat$test

###### CVgeneric
CVgeneric <- function(Data, Label, classifier, Kfold, Metric, Seeds, BlockNumber){
  # Data: Whole training data including all the candidate features
  # Label: The corresponding training labels
  # method: The classifier which is implemented
  # Kfold: Number of fold K
  # Metric: Loss function
  # Seeds: The chosen random seeds
  ###### General function to split CloudData
  CVDataGenerator = function(CloudData,BlockNum){
    set.seed(2019)
    RowNum = ColNum = sqrt(BlockNum)
    xQuantile = quantile(x = CloudData$x,probs = seq(0,1,length.out = ColNum))
    yQuantile = quantile(x = CloudData$y,probs = seq(0,1,length.out = RowNum))
    BlockList = list()
    for(i in 1:RowNum){
      for(j in 1:ColNum){
        ### Include last point to avoid losing info
        xRange = c(xQuantile[i],xQuantile[i+1]-sum(i<RowNum))
        yRange = c(yQuantile[j],yQuantile[j+1]-sum(j<ColNum))
        PixelChoice = intersect(which(xRange[1]<=CloudData$x & CloudData$x <=xRange[2]),which(yRange[1]<=CloudData$y & CloudData$y <=yRange[2]))
        BlockList[[(i-1)*ColNum+j]] = CloudData[PixelChoice,]
      }
    }
    ### Split to ten folds as list
    CVData = list()
    ValSize = ceiling(BlockNum*0.1)
    TrainSize = BlockNum - ValSize
    Shuffle = sample(1:BlockNum)
    for(j in 1:10){
      if(j<10){
        ValIndex = Shuffle[((j-1)*ValSize+1):(j*ValSize)]
      }else{
        ValIndex = Shuffle[((j-1)*ValSize+1):BlockNum]
      }
      TrainData = BlockList[[Shuffle[-ValIndex][1]]]
      for(i in Shuffle[-ValIndex][-1]){
        TrainData = rbind(TrainData,BlockList[[i]])
      }
      ValData = BlockList[[Shuffle[ValIndex][1]]]
      for(i in Shuffle[ValIndex][-1]){
        ValData = rbind(ValData,BlockList[[i]])
      }
      CVData[[j]] = list(train = TrainData,val = ValData)
    }
    return(CVData)
  }
  df = cbind(Label, Data)
  CVData = CVDataGenerator(df, BlockNumber)
  
  
  FoldAccuracies = rep(0,10)
  print('CVData has generated.')
  
  #for(k in 1:10){
    #print('Start training')
    #time.start = Sys.time()
    #Model <- svm(Label ~., data = CVData[[k]]$train, kernel = 'linear')
    #time.end = Sys.time()
    #print(time.end - time.start)
    #print('Start prediction')
    #time.start = Sys.time()
    #Prediction <- predict(Model, newdata = CVData[[k]]$val)
    #time.end = Sys.time()
    #print(time.end - time.start)
    #FoldAccuracies[k] = sum(Prediction==CVData[[k]]$val$Label)/dim(CVData[[k]]$val)[1]
   # print(k)
  #}
  
  for(k in 1:10){
    print('Start training')
    time.start = Sys.time()
    Model = train(Label~., data=CVData[[k]]$train, method = 'qda', family = 'binomial', metric = Metric)
    Prediction = predict(Model, CVData[[k]]$val)
    cvge.qda = as.numeric(Prediction) - 1    
    Model = train(Label~., data=CVData[[k]]$train, method = 'svmRadial', family = 'binomial', metric = Metric)
    Prediction = predict(Model, CVData[[k]]$val)
    cvge.svmR = as.numeric(Prediction) - 1
    pred.mix = 0.5*(cvge.qda) + 0.5*cvge.svmR 
    pred.mix.label = rep(0,length(pred.mix))
    pred.mix.label[pred.mix >= 0.5] = 1
    pred.mix.label[pred.mix < 0.5] = -1
    table.mix = table(pred.mix.label,CVData[[k]]$val$Label)
    FoldAccuracies[k] = sum(diag(table.mix))/sum(table.mix)
    print(FoldAccuracies[k])
    time.end = Sys.time()
    print(time.end - time.start)
    print(k)
  }

  FinalAccuracy = mean(FoldAccuracies)
  Results = list(FinalAccuracy = FinalAccuracy, FoldAccuracies = FoldAccuracies)
  return(Results)
}

# 3 Modeling 
#### Merge train and validation set
datTrain = rbind(datTrain,datVal)
### Test multivariate normality
set.seed(2019)
MVNindex = sample(1:dim(datTrain)[1], size = 5000, replace = F)
datMVNtest = datTrain[MVNindex,]
result <- mvn(data = datMVNtest[,4:6], mvnTest = 'mardia',
               univariatePlot = 'histogram')
cov(datTrain[datTrain$label == 1,4:6])
cov(datTrain[datTrain$label == -1,4:6])

#### lda
model.lda <- lda(label ~ NDAI + SD + CORR, data = datTrain)
pred.lda <- predict(model.lda, newdata = datTest)
table.lda <- table(datTest$label,pred.lda$class)
accuracy.lda <- sum(diag(table.lda))/sum(table.lda)
accuracy.lda
cv.lda = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'lda', 
                   Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 225)
cv.lda.2 = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'lda', 
                     Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 625)
cv.lda.2

#### qda
model.qda <- qda(label ~ NDAI + SD + CORR, data = datTrain)
pred.qda <- predict(model.qda, newdata = datTest)
table.qda <- table(datTest$label,pred.qda$class)
accuracy.qda <- sum(diag(table.qda))/sum(table.qda)
accuracy.qda
cv.qda = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'qda', 
                   Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 225)
cv.qda.2 = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'qda', 
                     Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 625)
cv.qda.2

#### logistic regression
model.logit <- glm(label ~ NDAI + SD + CORR, data = datTrain,
                   family = 'binomial')
pred.logit <- predict(model.logit, newdata = datTest, type = 'response')
pred.logit.label = rep(0,length(pred.logit))
pred.logit.label[pred.logit < 0.5] = -1
pred.logit.label[pred.logit >= 0.5] = 1
table.logit <- table(datTest$label,pred.logit.label)
accuracy.logit <- sum(diag(table.logit))/sum(table.logit)
accuracy.logit
cv.logit = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'glm', 
                     Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 225)
cv.logit.2 = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'glm', 
                       Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 625)
cv.logit.2

#### Linear SVM
model.svm <- svm(label ~ NDAI + SD + CORR, data = datTrain, kernel = 'linear')
pred.svm <- predict(model.svm, newdata = datTest, decision.values = TRUE)
table.svm <- table(datTest$label,pred.svm)
accuracy.svm <- sum(diag(table.svm))/sum(table.svm)
accuracy.svm
plot.svm(model.svm)
cv.svm = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'svmLinear', 
                   Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 225)
cv.svm
cv.svm.2 = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'svmLinear', 
                     Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 625)
cv.svm.2

#### Kernel SVM with radial basis
model.svm.k1 <- svm(label ~ NDAI + SD + CORR, data = datTrain, kernel = 'radial',
                    gamma = 1/3, cost = 1)
pred.svm.k1 <- predict(model.svm.k1, newdata = datTest, decision.values = TRUE)
table.svm.k1 <- table(datTest$label,pred.svm.k1)
accuracy.svm.k1 <- sum(diag(table.svm.k1))/sum(table.svm.k1)
accuracy.svm.k1
cv.svm.k1 <- CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'svmRadial', 
                       Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 225)
cv.svm.k1
cv.svm.k1.2 = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'svmRadial', 
                     Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 625)
cv.svm.k1.2

# ROC plot
### lda
df.lda <- data.frame(predictions = pred.lda$posterior[,2], labels = datTest$label)
p.lda <- ggplot(df.lda, aes(m = predictions, d = labels))+ geom_roc(cutoffs.at = 0.5) +
  style_roc(theme = theme_grey) + 
  annotate("text", label = 'AUC=0.9420', x = 0.75, y = 0.25, size = 5) + 
  labs(title = 'ROC curve of LDA') +
  theme(plot.title = element_text(hjust = 0.5, size = 15))
auc(df.lda$labels,df.lda$predictions)

### qda
df.qda <- data.frame(predictions = pred.qda$posterior[,2], labels = datTest$label)
p.qda <- ggplot(df.qda, aes(m = predictions, d = labels))+ geom_roc(cutoffs.at = 0.5) + 
  style_roc(theme = theme_grey) +
  annotate('text', label = 'AUC=0.9538', x = 0.75, y = 0.25, size = 5) + 
  labs(title = 'ROC curve of QDA') + 
  theme(plot.title = element_text(hjust = 0.5, size = 15))
auc(df.qda$labels,df.qda$predictions)

### logit
df.logit <- data.frame(predictions = pred.logit, labels = datTest$label)
p.logit <- ggplot(df.logit, aes(m = predictions, d = labels))+ geom_roc(cutoffs.at = 0.5) + 
  style_roc(theme = theme_grey) +
  annotate('text', label = 'AUC=0.9417', x = 0.75, y = 0.25, size = 5) + 
  labs(title = 'ROC curve of Logistic Regression') + 
  theme(plot.title = element_text(hjust = 0.5, size = 15))
auc(df.logit$labels,df.logit$predictions)

### linear svm
pred.svm.prob = 1 - exp(attr(pred.svm, 'decision.values')[,1]) / (1 + exp(attr(pred.svm, 'decision.values')[,1]))
df.svm <- data.frame(predictions = pred.svm.prob, labels = datTest$label)
p.svm <- ggplot(df.svm, aes(m = predictions, d = labels))+ geom_roc(cutoffs.at = 0.5) + 
  style_roc(theme = theme_grey) +
  annotate('text', label = 'AUC=0.9423', x = 0.75, y = 0.25, size = 5) + 
  labs(title = 'ROC curve of Linear SVM') + 
  theme(plot.title = element_text(hjust = 0.5, size = 15))
auc(df.svm$labels,df.svm$predictions)

### kernel svm
pred.svm.k1.prob = 1 - exp(attr(pred.svm.k1, 'decision.values')[,1]) / (1 + exp(attr(pred.svm.k1, 'decision.values')[,1]))
df.svm.k1 <- data.frame(predictions = pred.svm.k1.prob, labels = datTest$label)
p.svm.k1 <- ggplot(df.svm.k1, aes(m = predictions, d = labels))+ geom_roc(cutoffs.at = 0.5) + 
  style_roc(theme = theme_grey) +
  annotate('text', label = 'AUC=0.9608', x = 0.75, y = 0.25, size = 5) + 
  labs(title = 'ROC curve of Kernel SVM') + 
  theme(plot.title = element_text(hjust = 0.5, size = 15))
auc(df.svm.k1$labels,df.svm.k1$predictions)

ggarrange(p.lda,p.qda,p.logit, p.svm, p.svm.k1, ncol = 3, nrow = 2)

### Assess models by confusion matrix
f1score <- function(m){
  TP = m[2,2]
  TN = m[1,1]
  FP = m[1,2]
  FN = m[2,1]
  prec = TP / (TP + FP)
  rec = TP / (TP + FN)
  f1 = 2 * prec * rec / (prec + rec)
  return(f1)
}
# lda
table.lda
f1score(table.lda)
  
# qda
table.qda
f1score(table.qda)

# logistic regression
table.logit
f1score(table.logit)

# linear svm
table.svm
f1score(table.svm)

# kernel svm
table.svm.k1
f1score(table.svm.k1)

# 4 Diagnostics
## (a)
### increase size of training set
N = nrow(datTrain)
accuracy.diag = rep(0,20)
intercept.diag = rep(0,20)
coef.diag = matrix(0,nrow = 20, ncol = 3)
for(i in 1:20){
  datTrainsub <- datTrain[1:floor(i / 20 * N),]
  model <- svm(label ~ NDAI + SD + CORR, data = datTrainsub, kernel = 'radial',
               gamma = 1/3, cost = 1)
  pred <- predict(model, newdata = datTest)
  tabl <- table(datTest$label,pred)
  accuracy.diag[i] <- sum(diag(tabl))/sum(tabl)
  intercept.diag[i] <- model$rho
  coef.diag[i,] = t(model$coefs) %*% model$SV
  print(i)
}
df.coef = data.frame(value = c(coef.diag[,1],coef.diag[,2],
                               coef.diag[,3],intercept.diag),
                     percentage = rep(1:20/20,4),
                     coef = factor(c(rep(1,20),rep(2,20),rep(3,20),
                               rep(4,20))))
ggplot(df.coef) + geom_line(aes(x = percentage, y = value, col = coef)) + 
  labs(x = 'percentage of training data', y = 'values',
       title = 'Diagnostic Plot of Coefficient Convergence') +
  theme(plot.title = element_text(hjust = 0.5, size = 15)) +
  scale_color_discrete(name = 'coef', labels = c('NDAI','SD','CORR','intercept'))
ggplot() + geom_line(aes(x = 1:20/20, y = accuracy.diag)) + ylim(c(0.8,1)) +
  labs(x = 'percentage of training data', y = 'test accuracy', 
       title = 'Diagnostic Plot of Accuracy Convergence') +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

## (b)
wrong.index.train = which(model.svm.k1$fitted != datTrain$label)
wrong.index.test = which(pred.svm.k1 != datTest$label)
wrong.label.train = rep('correct',nrow(datTrain))
wrong.label.train[wrong.index.train] = 'wrong'
wrong.label.test = rep('correct',nrow(datTest))
wrong.label.test[wrong.index.test] = 'wrong'
wrong.train = datTrain[wrong.index.train,]
wrong.test = datTest[wrong.index.test,]
hist(wrong.train$NDAI)
hist(datTrain$NDAI)
hist(wrong.test$NDAI)
# NDAI appears to be an interesting feature.
ggplot() + geom_histogram(aes(x = datTrain$NDAI, fill = wrong.label.train), 
                          col = 'black', bins = 30) + 
  labs(x = 'NDAI', y = 'count', title = 'Histogram of NDAI of training set') +
  theme(plot.title = element_text(hjust = 0.5, size = 15)) +
  scale_fill_discrete(name = 'classification')
ggplot() + geom_histogram(aes(x = datTest$NDAI, fill = wrong.label.test), 
                          col = 'black', bins = 30) +
  labs(x = 'NDAI', y = 'count', title = 'Histogram of NDAI of test set ') +
  theme(plot.title = element_text(hjust = 0.5, size = 15)) +
  scale_fill_discrete(name = 'classification')
  

ggplot() + geom_point(aes(x = c(datTrain$x,datTest$x), y = c(datTrain$y,datTest$y), 
                          col = c(wrong.label.train,wrong.label.test))) +
  scale_color_discrete(name = 'classification') +
  labs(x = 'x coordinate', y = 'y coordinate', 
       title = 'Plot of correct/wrong classification w.r.t coordinates') +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

## (c)
### Mixture of model
pred.logit.label.prime = pred.logit.label + 1
pred.logit.label.prime[pred.logit.label.prime == 0] = 1
mix.lda = as.numeric(pred.lda$class) - 1
mix.qda = as.numeric(pred.qda$class) - 1
mix.logit = pred.logit.label.prime - 1
mix.svm = as.numeric(pred.svm) - 1
mix.svm.k1 = as.numeric(pred.svm.k1) - 1
pred.mix = 0.5*(mix.qda) + 0.5*mix.svm.k1 
pred.mix.label = rep(0,length(pred.mix))
pred.mix.label[pred.mix >= 0.5] = 1
pred.mix.label[pred.mix < 0.5] = -1
table.mix = table(pred.mix.label,df.svm$labels)
accuracy.mix = sum(diag(table.mix))/sum(table.mix)
accuracy.mix
ggplot()+ geom_roc(aes(m = pred.mix.label, d = datTest$label), cutoffs.at = 0.5) + 
  style_roc(theme = theme_grey) +
  annotate('text', label = 'AUC=0.9417', x = 0.75, y = 0.25, size = 5) + 
  labs(title = 'ROC curve of Logistic Regression') + 
  theme(plot.title = element_text(hjust = 0.5, size = 15))
cv.mixture = CVgeneric(Data = datTrain[,c(1:2,4:6)], Label = datTrain$label, classifier = 'glm', 
                       Kfold = 10, Metric = 'Accuracy', Seeds = 2019, BlockNumber = 225)

