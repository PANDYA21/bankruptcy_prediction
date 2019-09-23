source('dependencies.R')

# read all .arff files in data folder
arff_files <- list.files(path = 'data', pattern = '.arff', full.names = T)
# alldat <- rbindlist(lapply(arff_files[3:5], function(i_file) {
#   dat <- read.arff(i_file)
#   dat$year <- as.numeric(gsub('year', '', gsub('data/', '', gsub('.arff', '', i_file))))
#   return(dat)
# }))
alldat <- read.arff(arff_files[3])

# ranomForest for varImp and inital accuracy
thisdat <- na.spline(alldat[, -c(65)])
thisdat <- data.table(thisdat, mclass = factor(alldat$class))
set.seed(123)
model_rf <- randomForest(mclass~., data = thisdat, ntrees = 100, mtry = 100)
model_rf
varImpPlot(model_rf)

# par
require('parallel')
ntrees <- seq(50, 500, 50)
model_rfs <- mclapply(ntrees, function(ntree) {
  set.seed(123)
  thismodel <- randomForest(mclass~., data = thisdat, ntree = ntree, mtry = 100)
  return(list(
    ntree = ntree,
    model = thismodel
  ))
}, mc.cores = 4)

# 
save.image(paste0(Sys.Date() - 1, '.RData'))

# recalls
evals <- rbindlist(lapply(model_rfs, function(x) {
  cf <- x$model$confusion
  TP <- cf[2,2]
  TN <- cf[1,1]
  FP <- cf[1,2]
  FN <- cf[2,1]
  prec <- TP / (TP + FP)
  rec <- TP / (TP + FN)
  return(list(
    ntree = x$ntree,
    precision = prec,
    recall = rec,
    f1 = (prec + rec) / 2
  ))
}))

# plot(x = evals$ntree, y = evals$f1, type = 'b')

# viz
require('ggplot2')
ggplot(evals) + geom_point(
  aes(x = ntree, y = f1, color = 'f1')
) + geom_line(
  aes(x = ntree, y = f1, color = 'f1')
) + geom_point(
  aes(x = ntree, y = precision, color = precision)
) + geom_line(
  aes(x = ntree, y = precision, color = precision)
)

ggplot(melt(evals, id.vars = 'ntree'), aes(x = ntree, y = value, color = variable)) + 
  geom_point() + geom_line()


# best model
themodel <- model_rfs[[length(ntrees)]]
data.frame(themodel$model$importance)[order('MeanDecreaseGini'),]

