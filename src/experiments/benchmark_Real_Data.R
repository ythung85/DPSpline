# Load the libraries
library(earth)
library(mgcv)
library(readr)
library(dplyr)
library(mda)
library(Metrics)
library(pROC)


read_and_split_csv <- function(file_path) {
  # Read the CSV file into a data frame
  data <- read.csv(file_path)[, -1]
  
  # Ensure the data frame has at least two columns
  if (ncol(data) < 2) {
    stop("The CSV file must have at least two columns to split into X and y.")
  }
  
  # Extract the last column as the target variable (y)
  # using the 'ncol(data)' to dynamically select the last column index
  y <- data[, ncol(data)]
  
  # Extract all columns except the last one as features (X)
  # using a negative index to drop the last column
  X <- data[, -ncol(data)]
  
  data.frame(X = X, target = y)
}

criterion <- function(task, y_true, y_pred){
  if (task == "regression"){
    sqrt(mean((y_pred - y_true)^2))
  }
  else if (task == "classification"){
    auc(y_true, y_pred)
  }
}

run_exp <- function(data.name, task){
  
  Train_data <- read_and_split_csv(paste0("../real-data/",data.name, '_Train.csv'))
  Test_data <- read_and_split_csv(paste0("../real-data/",data.name, '_Test.csv'))
  
  numeric_vars <- colnames(Train_data)[1:length(Train_data)-1]
  # P-Spline
  smooth_terms <- paste0("s(", numeric_vars, ", bs='ps', k=7)")
  formula_str <- paste("target ~", paste(smooth_terms, collapse=" + "))
  model_formula <- as.formula(formula_str)
  
  if (task == "regression"){
    family_class = gaussian()
  }
  else if (task == "classification"){
    family_class = binomial(link = "logit")
  }
  
  pspline_model <- gam(model_formula, 
                       family=family_class, 
                       method="REML", 
                       data=Train_data)
  
  mars_model <- earth(
    target ~ .,  
    data = Train_data,
    degree = 2,       
    #nfold = 5,
    glm = list(family = family_class)
  )
  
  ## Evaluation
  
  pred_pspline <- predict(pspline_model, Test_data)
  mars_pred <- predict(mars_model, newdata = Test_data, type = "response")
  
  print(paste("Case:", data.name))
  
  if (task == "regression"){
    rmse_pspline = criterion(task, Test_data$target, pred_pspline)
    rmse_mars = criterion(task, Test_data$target, mars_pred)
    print(paste("P-spline RMSE:", round(rmse_pspline, 4)))
    print(paste("MARS RMSE:", round(rmse_mars, 4)))
  }
  else if (task == 'classification'){
    auc_pspline = criterion(task, Test_data$target, pred_pspline)
    auc_mars = criterion(task, Test_data$target, mars_pred)
    print(paste("P-spline AUC:", round(auc_pspline, 4)))
    print(paste("MARS AUC:", round(auc_mars, 4)))
  }
}
    

run_exp('ca', 'regression')
run_exp('bike', 'regression')
run_exp('year', 'regression')
run_exp('churn', 'classification')


## Results - ms

MSPE1 = c(0.08789071, 0.04671669, 0.0497563,  0.04353564, 0.0897031,  0.09456228,
          0.08053329, 0.04760924, 0.06394239, 0.07043544, 0.03920984, 0.05582673, 0.04280233,
          0.04677242, 0.06229917, 0.04655605, 0.0478022,  0.05444681,
          0.06788949,0.02863338, 0.07294435, 0.2066886,  0.12960082, 0.07724191, 0.05850098,
          0.06726435, 0.1651987,  0.09013265, 0.06851254, 0.06989481, 0.06522442,
          0.1169534,  0.08059414, 0.12108645, 0.04927954, 0.05806023, 0.06233066,
          0.04090535, 0.0808391, 0.16113485, 0.06186919, 0.11361992, 0.1244841,  0.06164247, 0.02988391,
          0.04351306, 0.04126745, 0.34951481, 0.05396824, 0.05000911, 0.05036652,
          0.06577896, 0.09501135, 0.0606152,  0.09433988, 0.04751009, 0.02809549,
          0.07792237, 0.11442521, 0.03851106, 0.14778557, 0.06549504, 0.09397328, 0.14034732, 0.06425067,
          0.08734364, 0.30580682, 0.06182866, 0.25871885, 0.07586636, 0.05938038,
          0.04793745, 0.05134248, 0.04053504, 0.05740115, 0.04325992, 0.07246037,
          0.04634149, 0.07271943)
MSPE2 = c(0.04418093, 0.06025865, 0.02694539, 0.0369102,  0.03673492, 0.04599024,
          0.02485384, 0.03089103, 0.02310862, 0.01723621, 0.02619362, 0.01838488,
          0.01801152, 0.0416573,  0.03825931, 0.06462001, 0.01570718, 0.02753998,
          0.07052889,0.02590882, 0.03520589, 0.03956763, 0.02110755, 0.11038946, 0.02493041,
          0.02753543, 0.08192407, 0.04702399, 0.03245565, 0.02187757, 0.04849397,
          0.07476024, 0.08016218, 0.02161659, 0.07940299, 0.0549382,  0.02415986,
          0.02236336, 0.0797368, 0.02966724, 0.09411375, 0.31811076, 0.04659784, 0.03503364, 0.04137528,
          0.03671672, 0.04975988, 0.03134016, 0.11889104, 0.04545437, 0.08177851,
          0.01573456, 0.03608408, 0.02212329, 0.02785365, 0.06438754, 0.05386373,
          0.07217071, 0.03602503,0.2453942,  0.11245725, 0.02476866, 0.01796,    0.02192414, 0.02483978,
          0.10603166, 0.18070017, 0.03642648, 0.04946703, 0.0460574,  0.03924024,
          0.02979438, 0.06176965, 0.03145215, 0.02409641, 0.02434643, 0.03240944,
          0.04539537, 0.09475714)


mean(MSPE1)
var(MSPE1)**0.5
mean(MSPE2)
var(MSPE2)**0.5
