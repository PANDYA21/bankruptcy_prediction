## set CRAN repo
local({
  r <- getOption("repos")
  r["CRAN"] <- "http://cran.r-project.org"
  options(repos=r)
})

## custom require function that installs dependencies if not installed already and then requires them
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

## load libraries
#. to read arff files
require('foreign')
#. to impute
require('zoo')
#. data.frame upgrade
require('data.table')
#. RF
require('randomForest')
