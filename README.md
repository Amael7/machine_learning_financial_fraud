# machine_learning_financial_fraud

##### Use machine learning algorithms to analyse financial fraud dataset and make a tier list of them

--------------------------------------------------------------------------

There all the algorithm 

- **Unsupervised Learning**
	- **Association**
		- Apriori Algorithm
	- **Clustering**
		- K-Means
		- Hierarchical Clustering
		- Gaussian Mixture Models
- **Supervised Learning**
	- **Tree-Based Models**
		- Decision Tree
		- Random Forests
		- Gradient Boosting Regression
		- XGBoost
		- LightGBM Regressor
	- **Linear Models**
		- Linear Regression
		- Logistic Regression
		- Ridge Regression
		- Lasso Regression

algorithms_source_url : https://www.kaggle.com/code/arjunjoshua/predicting-fraud-in-financial-payment-services/notebook

--------------------------------------------------------------------------

##### The criteria that'll be used to determine which algorithms is the best for the data we'll analyse

One of the most important criteria is the data types that we'll be using.

All algorithm has a different purpose and it's pretty difficult to determine a tiers list because of that.

- Execution time 
- Error's rate
- Accuracy

# Top 3 Algorithm for that use case

1. Random Forest
	- Model Accuracy: 99.81%
	- Model Error Rate: 0.19%
    - Execution Time : 251.90 seconds
   	- Observations : Powerful, well-balanced algorithm capable of processing large amounts of complex data and delivering high prediction accuracy.
2. XGB Regressor
   - Model Accuracy: 99.80%
   - Model Error Rate: 0.20%
   - Execution Time : 330.22 seconds
   - Observations : Excellent performance but less accurate than random forest
3. Decision tree
   - Model Accuracy: 99.75%
   - Model Error Rate: 0.25%
   - Execution Time : 18.57 seconds
   - Observations : High accuracy and turnaround time, but higher error rate than others


--------------------------------------------------------------------------

# Tier List of all Algorithm

### Tier 1 - Recommended Algorithms:

1. **Ensemble Methods (Random Forest, XGBoost, LightGBM):** These algorithms are robust and can handle complex datasets with high performance.
    
2. **Neural Networks (Deep Learning):** Neural networks, especially deep architectures, can capture complex patterns in data but often require significant resources.
    
3. **Logistic Regression:** A simple yet effective model for binary classification.
    
4. **Support Vector Machines (SVM):** Can be effective for separating classes in a high-dimensional feature space.
    

### Tier 2 - Potential Alternatives:

5. **Lasso or Ridge Regression:** For regularization and feature selection.
    
6. **K-Nearest Neighbors (KNN):** Can be effective for detecting anomalies or local patterns.
    
7. **Isolation Forest:** Designed to detect anomalies, particularly useful for fraud detection.
    

### Tier 3 - Contextual Use:

8. **Bayesian Networks:** Can be effective when probabilistic assumptions are useful.
    
9. **Quadratic Discriminant Analysis (QDA):** When the distribution of classes may be nonlinear.
    
10. **Clustering Algorithms (K-Means, DBSCAN):** For detecting anomalies or unexpected patterns.
    

### Special Considerations:

11. **Isolation Forest:** Specifically designed for anomaly detection, can be effective for fraud.
    
12. **Autoencoders (Deep Learning):** Can be used to learn unsupervised representations and detect anomalies.