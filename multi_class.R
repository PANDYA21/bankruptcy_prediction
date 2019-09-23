source('dependencies.R')

# read all .arff files in data folder
arff_files <- list.files(path = 'data', pattern = '.arff', full.names = T)
alldat <- rbindlist(lapply(arff_files[3:5], function(i_file) {
  dat <- read.arff(i_file)
  dat$year <- as.numeric(gsub('year', '', gsub('data/', '', gsub('.arff', '', i_file))))
  return(dat)
}))

# prepare classes for multi class classification
alldat$mclass <- NA
alldat$mclass <- ifelse(alldat$class == 1, alldat$year, 0)
# View(alldat)

# ranomForest for varImp and inital accuracy
thisdat <- na.spline(alldat[, -c(65,66,67)])
thisdat <- data.table(thisdat, mclass = factor(alldat$mclass))
model_rf <- randomForest(mclass~., data = thisdat, ntrees = 50, mtry = 10)
model_rf
varImpPlot(model_rf)


#### decision trees
# require('ISLR')
require('rpart')
model_dt <- rpart(mclass~., data = thisdat, method = 'anova', )
model_dt
printcp(model_dt)
plotcp(model_dt)
summary(model_dt)
plot(model_dt)
text(model_dt)




