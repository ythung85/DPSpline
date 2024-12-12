library(MaxPro)
library(ggplot2)
# Link: https://cran.r-project.org/web/packages/MaxPro/MaxPro.pdf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Generate Candidate Design Points Randomly for Various Types of Factors #                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

CCCtrain<-CandPoints(N=5000,p_cont=9)
CCCtest<-CandPoints(N=250,p_cont=9)

# validation on testing data and select the parameter with highest eta value.

write.csv(CCCtrain,"/Users/a080528/Desktop/Github/DBspline/trainsample.csv", row.names = FALSE)
write.csv(CCCtest,"/Users/a080528/Desktop/Github/DBspline/testsample.csv", row.names = FALSE)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Generate the maximum projection (MaxPro) Latin hypercube #
#  design for continuous factors based on a simulated annea #
#  -ling algorithm                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

obj<-MaxProLHD(n = 10, p = 3)
obj$Design


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                         Kriging                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

library(rkriging)
X = read.csv('/Users/a080528/Desktop/Github/DBspline/Layer-wise optimization/output_feature.csv')
y = read.csv('/Users/a080528/Desktop/Github/DBspline/Layer-wise optimization/output_eta.csv')
X = as.matrix(X)
y = as.matrix(y)


testX = read.csv('/Users/a080528/Desktop/Github/DBspline/Layer-wise optimization/test_output_feature.csv')
testy = read.csv('/Users/a080528/Desktop/Github/DBspline/Layer-wise optimization/test_output_eta.csv')
testX = as.matrix(testX)
testy = as.matrix(testy)

model <- Fit.Kriging(X, y, interpolation=TRUE, fit=TRUE, model="OK",
                     kernel.parameters=list(type="Gaussian"))

pred <- Predict.Kriging(model, testX)
pred_eta = pred$mean
pred_std = pred$sd

eta_max = pred_eta[which.max(pred_eta)]
sd_eta_max = pred_std[which.max(pred_eta)]
beta_max = 1.6459 # mapping from test data

sdSt <- function(t, e, b, sde){
  dSt <- (b*t**(b-1)/e**(b))*exp(-(t/e)**b)
  sqrt((dSt**2)*(sde**2))
}

st <- function(t,e,b){
  exp(-(t/e)**b)
}


sdeSt <- function(t, e, b, sde){
  dSt <- exp(-(t/exp(e))**b)*(-t**b)*(-b)* exp(-b*e)
  sqrt((dSt**2)*(sde**2))
}

sdst <- function(t,e,b){
  exp(-(t/exp(e))**b)
}

t <- seq(0, 150, by = 0.1)

## without log-transformation
stv = st(t, eta_max, beta_max)
sdstv = sdSt(t, eta_max, beta_max, sd_eta_max)

## with log-transformation
stv = sdst(t, eta_max, beta_max)
sdstv = sdeSt(t, eta_max, beta_max, sd_eta_max)

dft <- data.frame (
  time = t,
  vst = stv,
  uvst = stv+1.96*sdstv,
  lvst = stv-1.96*sdstv
)

ggplot(data = dft, aes(x = t, y = vst)) + 
  geom_ribbon(aes(ymin = lvst, ymax = uvst), fill = "deepskyblue", alpha = 0.2) + geom_line(aes(y = vst), color = "white", size = 1.5) +geom_point() + 
  ggtitle("Regression Line, 95% Confidence and Prediction Bands")


ggplot(dft, aes(t, vst)) +    
  geom_point()+ 
  # geom_ribbon function is used to add confidence interval 
  geom_ribbon(aes(ymin = lvst, ymax = uvst),  
              alpha = 0.2)
