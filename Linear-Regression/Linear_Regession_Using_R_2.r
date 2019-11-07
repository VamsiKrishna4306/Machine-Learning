setwd("D:/Bdat_Course_Material/Semister 2/Business Intelligence/Data Sets")
CompVenData = read.csv(file = "Computer_Data.csv")
View(CompVenData)
nrow(CompVenData)
set.seed(777)
RowNumbers = sample(1:nrow(CompVenData), 0.7*nrow(CompVenData))
head(RowNumbers)
CompVenTrainData = CompVenData[RowNumbers, ]
CompVenTestData = CompVenData[-RowNumbers, ]
nrow(CompVenTrainData)
nrow(CompVenTestData)
colnames(CompVenTrainData)
mod1 = lm(formula = price ~ speed + hd + ram + screen + ads + trend, data = CompVenTrainData)
plot(mod1$fitted.values, mod1$residuals, xlab = "Predicted Values", ylab = "Error")
abline(a=0, b=0)

