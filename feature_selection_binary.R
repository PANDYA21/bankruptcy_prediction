source('dependencies.R')

# read all .arff files in data folder
arff_files <- list.files(path = 'data', pattern = '.arff', full.names = T)

anss <- lapply(arff_files[3:5], function(i_file) {
  dat <- read.arff(i_file)
  # ranomForest for varImp and inital accuracy
  thisdat <- na.spline(dat[, -c(65)])
  thisdat <- data.table(thisdat, class = factor(dat$class))
  model_rf <- randomForest(factor(class)~., data = thisdat, ntrees = 1500, mtry = 3)
  # importance
  imp <- data.frame(model_rf$importance)
  imp$x <- rownames(imp)
  imp <- data.table(imp)
  imp <- imp[order(-MeanDecreaseGini)]
  imp$x <- factor(imp$x, levels = imp$x)
  return(list(
    model_rf = model_rf
  ))
})


# prepare importance data to visualize
imps <- lapply(anss, function(x) {
  imp <- data.frame(x$model_rf$importance)
  imp$x <- rownames(imp)
  imp <- data.table(imp)
  imp <- imp[order(-MeanDecreaseGini)]
  imp$x <- factor(imp$x, levels = imp$x)
  return(imp)
})

j = 3
for (i in 1:length(imps)) {
  imps[[i]]$year <- paste0('year ', j)
  j <- j+1
}

imps <- rbindlist(imps)

# visualize
require('ggplot2')
plt <- ggplot(imps) + geom_bar(
  aes(x = x, y = MeanDecreaseGini),
  stat = 'identity'
) + theme(
  axis.text.x = element_text(angle = 90)
) + ggtitle(label = 'Variable Importance') + facet_wrap(~year, ncol = 1)

ggsave('varimp.png', plt, width = 12, height = 8, units = 'in', dpi = 600)


