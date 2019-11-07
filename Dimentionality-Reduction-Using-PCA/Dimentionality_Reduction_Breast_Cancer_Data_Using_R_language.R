
setwd("D:/Bdat_Course_Material/Semister 2/Business Intelligence/Assignment2")
CancerData = read.csv("breast-cancer-data.csv",header=TRUE)
# View(CancerData)
nrow(CancerData)
CancerData_features <- CancerData[,3:32]
View(CancerData_features)
pc <- prcomp(CancerData_features,cor=TRUE,score=TRUE)
summary(pc)

