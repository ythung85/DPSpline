library(MaxPro)
library(ggplot2)
library(rkriging)

# Link: https://cran.r-project.org/web/packages/MaxPro/MaxPro.pdf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Generate Candidate Design Points Randomly for Various Types of Factors #                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

CCCtrain<-CandPoints(N=5000,p_cont=9)
CCCtest<-CandPoints(N=250,p_cont=9)

# validation on testing data and select the parameter with highest eta value.

write.csv(CCCtrain,"./dataset/trainsample.csv", row.names = FALSE)
write.csv(CCCtest,"./dataset/testsample.csv", row.names = FALSE)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                         Kriging                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
X = read.csv('./dataset/output_feature.csv')
y = read.csv('./dataset/output_eta.csv')
X = as.matrix(X)
y = as.matrix(y)


testX = read.csv('./dataset/test_output_feature.csv')
testy = read.csv('./dataset/test_output_eta.csv')
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
  ggtitle("Regression Line, 95% Confidence and Prediction Bands") + ylab('S(t)')

