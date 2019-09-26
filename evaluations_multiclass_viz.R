source('dependencies.R')
require('ggplot2')

evals <- fread('evaluations_multiclass.csv')

reshaped <- melt(
  evals, 
  id.vars = c('model'), 
  measure.vars = c(
    "accuracy_train", "precision_train", "recall_train", "f1_train", 
    "accuracy_test", "precision_test", "recall_test", "f1_test"))

reshaped$variable <- as.character(reshaped$variable)
separated <- rbindlist(lapply(reshaped$variable, function(x) {
  return(as.list(unlist(strsplit(x, '_'))))
}))
names(separated) <- c('Measure', 'Type')
reshaped <- cbind(reshaped, separated)

reshaped$Type <- factor(reshaped$Type, levels = c('train', 'test'))
reshaped$Measure <- factor(reshaped$Measure, levels = c("accuracy", "precision", "recall", "f1"))

plt <- ggplot(reshaped) + geom_bar(
  aes(x = factor(1), y = value, fill = model),
  width = 0.5,
  stat = 'identity',
  position = 'dodge'
) + facet_wrap(Measure~Type, ncol = 2) + ggtitle(
  label = 'Evaluation of Multi-Class Classification', 
  subtitle = ' '
) + theme(
  axis.text.x = element_blank()
) + xlab('') + ylab('')
plt

ggsave(filename = 'evaluations_multiclass.pdf', plot = plt, width = 7, height = 10, units = 'in')
