**Results** : We obtained a accuracy of 98.32 which is quite good, the used algorithm was Random Forest. The order in which the algorithm presented the results are : 
	Random Forest : 98.32  
	Decision Tree :	98.32  
	KNN	: 82.38  
	Logistic Regression :	80.81  
	Naive Bayes	: 79.24  
	Linear SVC	: 78.34  
	Stochastic Gradient Decent	: 70.48  
	Support Vector Machines	: 68.24  
	Perceptron	: 65.66  

**Reason for choosing Random Forest** : Its because it takes care of cross validataion and it also is giving the best accuracy among the other models. Hence choosing it will be the best way. 

**Problems faced** : Conversion of Categorical Data into Numerical Data, Filling the missing data. I tackled them using the conversion by TRANSFORMATION using sklearn tools. Also to tackle missing datas, whereever the missing data was less it was intuitive to remove those entries and also replaced  few datas using mean values from the dataset. For age the better solution was to replace it with median. 

**Conculsion** : Clearly we can see that both Decision Tree and Random Forest can be implemented and the score recieved is also pretty decent.
One might has a better chance if she is a woman or a child also we have a better chance of surviving if we have a higher class ticket than if we had a third class ticket. In comparison to Southampton or Queenstown, a man is more likely to live if he embarks in Cherbourg. If we travel with 1 or 3 persons instead of 0 or more than 3, our chances of survival increase.
