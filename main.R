setwd("C:/Users/Sagar/OneDrive/Documents/Learning/Higgs")
#check for, install and load packages
if(!require(xgboost)) install.packages("xgboost",repos=c("http://rstudio.org/_packages", "http://cran.rstudio.com"))
library(xgboost)
if(!require(doParallel)) install.packages("doParallel")
library(doParallel)
if(!require(methods)) install.packages("methods")
library(methods)
if(!require(caret)) install.packages("caret")
library(caret)

#register parallel backend
cl<-makeCluster(detectCores())
registerDoParallel(cl)
getDoParWorkers()

#declare test size (needed for scaling weights)
testsize<-550000

#read training data
xtrain<-read.csv("training.csv", header=TRUE)

#convert label to number because xgboost works with numbers for class labels
xtrain[33]<-xtrain[33]=="s"
label<- as.numeric(xtrain[[33]])

#store training data into a matrix (as xgb.Dmatrix requires a matrix object)
data<-as.matrix(xtrain[2:31])

#scale weights for each observation (this is important if ams is the metric used)
weight<-as.numeric(xtrain[[32]])*testsize/length(label)

#find class weights
sumpos<-sum(weight*(label==1.0))
sumneg<-sum(weight*(label==0.0))

#create xgb.Dmatrix object for training
xgmat<-xgb.DMatrix(data, label=label, weight=weight, missing=-999.0)

#create matrix to store tuning values
etatunevec=c(0.01,0.05,0.1)
maxdtunevec=c(5,6,7,8,9,10)
tuneresults<-matrix(0,length(etatunevec)*length(maxdtunevec),4)

#train via crossvalidation over the tuning grid
for(i in 1:length(etatunevec)){
  for(j in 1:length(maxdtunevec)){
    
    #create parameter list
    param<-list("objective"="binary:logitraw", #logitraw because easier to rank before sigmoid has been applied
                "scale_pos_weight"=sumneg/sumpos, #class weights
                "bst:eta"=etatunevec[i],
                "bst:max_depth"=maxdtunevec[j],
                "eval_metric"="auc", #auc metric because obvious class imbalance
                "silent"=1,
                "nthread"=4)
    
    #create watchlist
    watchlist<-list("train"=xgmat)
    
    #declare number of trees
    nround=12/etatunevec[i]
    
    #implement cross validation
    model.cv<-xgb.cv(param, xgmat, nround, nfold=8, showsd=1, metrics="auc")
    tuneresults[(i-1)*length(maxdtunevec)+j,1]=etatunevec[i]
    tuneresults[(i-1)*length(maxdtunevec)+j,2]=maxdtunevec[j]
    tuneresults[(i-1)*length(maxdtunevec)+j,3]=model.cv$train.auc.mean[nround]
    tuneresults[(i-1)*length(maxdtunevec)+j,4]=model.cv$test.auc.mean[nround]
  
  }
}

#convert tuneresults to data frame
tuneresults<-data.frame(tuneresults)
names(tuneresults)=c("eta","max_depth","train_auc","test_auc")

#plot tuning results
ggplot(tuneresults, aes(x=max_depth,col=factor(eta)))+
    geom_line(aes(y=test_auc))+
    ggtitle("Test AUC for Different Parameters")

#choose best parameters
best_eta=tuneresults$eta[tuneresults$test_auc==max(tuneresults$test_auc)]
best_maxd=tuneresults$max_depth[tuneresults$test_auc==max(tuneresults$test_auc)]
print(paste("The best parameters are eta =", best_eta, "and max_depth =", best_maxd))


#train final model based on best parameters
#create parameter list
param<-list("objective"="binary:logitraw",
            "scale_pos_weight"=sumneg / sumpos,
            "bst:eta"=best_eta,
            "bst:max_depth"=best_maxd,
            "eval_metric"="auc",
            "silent"=1,
            "nthread"=4)

#create watchlist
watchlist<-list("train" = xgmat)

#declare number of trees
nround=12/best_eta

#train final model on best parameters
model<-xgb.train(param, xgmat, nround, watchlist)

#get the feature names
names<-dimnames(data)[[2]]

#find important features
importance_matrix<-xgb.importance(names, model=model)

#display 10 most important features
xgb.plot.importance(importance_matrix[1:10,])

#plot first decision tree (just out of curiousity)
xgb.plot.tree(feature_names=names, model=model, n_first_tree=1)

#read test data
xtest<-read.csv("test.csv", header=TRUE)
data<-as.matrix(xtest[2:31])
idx<-xtest[[1]]
xgmat<-xgb.DMatrix(data, missing=-999.0)

#predict using final model
ypred<-predict(model, xgmat)

#rank by pre-sigmoid scores (since the kaggle evaluation requires ranking)
rorder<-rank(ypred, ties.method="first")

#choose a threshold
threshold<-0.15

#apply threshold
ntop<-length(rorder)-as.integer(threshold*length(rorder))
plabel<-ifelse(rorder>ntop, 1, 0)

#store in format required by kaggle
outdata<-data.frame("EventId"=idx,
                "RankOrder"=rorder,
                "Class"=plabel)
write.csv(outdata, file="higgs.pred.csv", quote=FALSE, row.names=FALSE)

#THIS TUNED MDOEL PRODUCES AN AMS SCORE OF AROUND 3.7, WHICH IS AROUND RANK 80/2000 ON KAGGLE
#THIS CAN BE FURTHER IMPROVED, PERHAPS BY TRYING A SMALLER ETA (TO BE DONE LATER)
