require('xgboost')
require('caret')
require('e1071')


fitControl <- fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 2, search = "random")
model_xgb <- train(mclass~., data = thisdat, method = 'xgbTree', trControl = fitControl)
print(model_xgb)
model_xgb$results
model_xgb$modelInfo
varImp(model_xgb)

predict(model_xgb$finalModel, thisdat[,-65])
