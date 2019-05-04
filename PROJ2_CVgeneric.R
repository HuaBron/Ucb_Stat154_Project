# Generic customized CV function

CVgeneric <- function(Data, Label, classifier, Kfold, Metric, Seeds, BlockNumber){
  # Data: Whole training data including x,y coordinates and all the candidate features
  # Label: The corresponding training labels
  # method: The classifier which is implemented
  # Kfold: Number of fold K
  # Metric: Loss function
  # Seeds: The chosen random seeds
  # BlockNumber: The block number of split data
  
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
    for(j in 1:Kfold){
      if(j < Kfold){
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
  
  library(caret)
  library(MASS)
  library(kernlab)
  library(e1071)
  df = cbind(Label, Data)
  CVData = CVDataGenerator(df, BlockNumber)
  
  # ctrl = trainControl(method = 'repeatedcv', number = Kfold, savePredictions = 'all', classProbs = FALSE)
  
  FoldAccuracies = rep(0,Kfold)
  for(k in 1:Kfold){
    Model = train(Label~., data=CVData[[k]]$train, method = classifier, family = 'binomial', metric = Metric)
    Prediction = predict(Model, CVData[[k]]$val)
    FoldAccuracies[k] = sum(Prediction==CVData[[k]]$val$Label)/dim(CVData[[k]]$val)[1]
  }
  FinalAccuracy = mean(FoldAccuracies)
  Results = list(FinalAccuracy = FinalAccuracy, FoldAccuracies = FoldAccuracies)
  return(Results)
}
