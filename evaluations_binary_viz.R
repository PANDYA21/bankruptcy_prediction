source('dependencies.R')
require('ggplot2')

evals <- fread('evaluations_binary.csv')

reshaped <- melt(
  evals, 
  id.vars = c('year', 'model'), 
  measure.vars = c(
    "accuracy_train", "precision_train", "recall_train", 
    "f1_train", "auc_train", "accuracy_test", 
    "precision_test", "recall_test", "f1_test", "auc_test"))

reshaped$variable <- as.character(reshaped$variable)
separated <- rbindlist(lapply(reshaped$variable, function(x) {
  return(as.list(unlist(strsplit(x, '_'))))
}))
names(separated) <- c('Measure', 'Type')
reshaped <- cbind(reshaped, separated)

reshaped$Type <- factor(reshaped$Type, levels = c('train', 'test'))
dput(unique(reshaped$Measure))
reshaped$Measure <- factor(reshaped$Measure, levels = c("accuracy", "precision", "recall", "f1", "auc"))
reshaped$year <- as.numeric(gsub('year.arff', '', gsub('data/', '', reshaped$year)))
reshaped$horizon <- 5 - reshaped$year + 1

plt <- ggplot(reshaped) + geom_bar(
  aes(x = horizon, y = value, fill = model),
  width = 0.5,
  stat = 'identity',
  position = 'dodge'
) + facet_wrap(Measure~Type, ncol = 2) + ggtitle(
  label = 'Evaluation of Binary Classification', 
  subtitle = ' '
) + xlab('Forecast Horizon')
plt

ggsave(filename = 'evaluations_binary.pdf', plot = plt, width = 7, height = 10, units = 'in')
