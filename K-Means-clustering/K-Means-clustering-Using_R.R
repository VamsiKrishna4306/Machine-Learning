
setwd("D:/Bdat_Course_Material/Semister 2/Business Intelligence/Assignment 4")
data <-read.csv("driver-data.csv",header=T)
summary(data)
data_features = data[,2:3]


rng<-2:10 #K from 2 to 10
tries <-100 #Run the K Means algorithm 100 times
avg.totw.ss <-integer(length(rng)) #Set up an empty vector to hold all of points
for(v in rng){ # For each value of the range variable
  v.totw.ss <-integer(tries) #Set up an empty vector to hold the 100 tries
  for(i in 1:tries){
    k.temp <-kmeans(data_features,centers=v) #Run kmeans
    v.totw.ss[i] <-k.temp$tot.withinss#Store the total withinss
  }
  avg.totw.ss[v-1] <-mean(v.totw.ss) #Average the 10 total withinss
}
plot(rng,avg.totw.ss,type="b", main="Total Within SS by Various K",
     ylab="Average Total Within Sum of Squares",
     xlab="Value of K")

set.seed(76964057) 
k <-kmeans(data_features, centers=4)
k$centers 
table(k$cluster)
# install.packages("factoextra")
library(factoextra)
library(cluster)

fviz_cluster(k, data = data_features)