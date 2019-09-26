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
model_rf <- randomForest(mclass~., data = thisdat, ntrees = 50, mtry = 2)
model_rf
# varImpPlot(model_rf)


imp <- data.frame(model_rf$importance)
imp$x <- rownames(imp)
imp <- data.table(imp)
imp <- imp[order(-MeanDecreaseGini)]
imp$x <- factor(imp$x, levels = imp$x)
plt <- ggplot(imp) + geom_bar(
  aes(x = x, y = MeanDecreaseGini),
  stat = 'identity'
) + theme(
  axis.text.x = element_text(angle = 90)
) + ggtitle(label = 'Variable Importance')
ggsave('varimp_multic.pdf', plt, width = 12, height = 5, units = 'in', dpi = 600)



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




