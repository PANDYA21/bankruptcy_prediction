local({
  r <- getOption("repos")
  r["CRAN"] <- "http://cran.r-project.org"
  options(repos=r)
})

# custom require function that installs dependencies if not installed already and then requires them
require <- function(...) {
  success <- base::require(...)
  if (!success) {
    tryCatch(
      expr = {
        install.packages(...)
        base::require(...)
      },
      error = function(e) {
        e
      },
      finally = gc() # garbage collection
    )
  }
}


require('foreign')
require('zoo')
require('randomForest')
require('data.table')

dat <- read.arff('data/1year.arff')
print(head(dat))


fit <- randomForest(factor(class)~., data = dat[complete.cases(dat),])
fit
varImpPlot(fit)

fit2 <- randomForest(factor(class)~., data = na.locf(dat))
fit2
varImpPlot(fit2)

dat3 <- na.approx(dat)
dat3 <- dat3[complete.cases(dat3),]
fit3 <- randomForest(factor(class)~., data = dat3)
fit3
varImpPlot(fit3)

fit4 <- randomForest(factor(class)~., data = na.spline(dat))
fit4
varImpPlot(fit4)


dat5 <- data.frame(na.locf(apply(dat[, -ncol(dat)], 2, scale)), class = dat$class)
fit5 <- randomForest(factor(class)~., data = dat5)
fit5
varImpPlot(fit5)



normalize <- function(vect) {
  return((vect - min(vect, na.rm = T))/(max(vect, na.rm = T) - min(vect, na.rm = T)))
}
dat6 <- data.frame(na.locf(apply(dat[, -ncol(dat)], 2, normalize)), class = dat$class)
fit6 <- randomForest(factor(class)~., data = dat6, ntrees = 1000, mtry = 10)
fit6
varImpPlot(fit6)




#########
arffs <- list.files(path = 'data', pattern = '.arff', full.names = T)

alldat <- rbindlist(lapply(arffs, function(fil) {
  dat <- read.arff(fil)
  dat$year <- as.numeric(gsub('year', '', gsub('data/', '', gsub('.arff', '', fil))))
  return(dat)
}))

normalize <- function(vect) {
  return((vect - min(vect, na.rm = T))/(max(vect, na.rm = T) - min(vect, na.rm = T)))
}
alldat2 <- data.frame(na.locf(apply(dat[, -ncol(dat)], 2, normalize)), class = dat$class)
fit7 <- randomForest(factor(class)~., data = alldat2, ntrees = 1000, mtry = 10)
fit7
varImpPlot(fit7)




##### dims
lapply(arffs, function(fil) {
  dat <- read.arff(fil)
  return(dim(dat))
})




