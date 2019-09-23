multi-class classification
or multiple binary classification models

bankruptcy prediction

## multiclass classification
models: 1 model
classes: 
1. will not bankrupt (no)
2. will bankrupt in 1 year (one)
3. will bankrupt in 2 years (two)
4. will bankrupt in 3 years (three)

## binary classification
models: 3 models
1. model 1: predicting bankruptcy after 1 year
2. model 1: predicting bankruptcy after 2 year
3. model 1: predicting bankruptcy after 3 year
classes: 
1. will not bankrupt
2. will bankrupt
final classification classes (ensemble):
1. will not bankrupt (model1 == 0 & model2 == 0 & model3 == 0)
2. will bankrupt in 1 year (model1 == 1)
3. will bankrupt in 2 years (model2 == 1)
4. will bankrupt in 3 years (model3 == 1)
