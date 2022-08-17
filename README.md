# My Data  Scientific Work

# [Research: Implementation of TFIDF vectorizer from scratch](https://github.com/PravinRedoc/Research/blob/main/TFIDF_Implementation.ipynb)
* Term Frequency/Inverse Document Frequency is the most popular text vectorization technique widely used in the text based analytics industry
* Implemented TFIDF vectorizer from scratch using numpy and pandas libraries.
* Deeper and Simple understanding of how TFIDF works internally.
* Comparision between actual library methods and implementation from scratch.


# [Research: Implementation of major Performance metrics used for Model Evaluation](https://github.com/PravinRedoc/Research/blob/main/Performance_metrics_Implementation.ipynb) 
* Implementaion of all the major and industry standard preformance metrics used to evaluate models such as confusion matrix, F1 score,AUC Score,MSE, MAPE, R^2
* Deeper understanding of how all these metrics work internally
* Comparision with sklearn metrics

# [Research: Implementation of K-Fold cross validation with RandomSearchCV from scratch](https://github.com/PravinRedoc/Research/blob/main/K-Fold-KNN.ipynb) 
* Implementaion of K-Fold cross validation using RandomSearchCV(Randomization using uniform distribution) from scratch
* K-NN classifer to build a model with cross validation
* Comparision with sklearn and visualizing the decision boundary.

# [Research: Implementation of SGD Classifier with Logloss and L2 regularization Using SGD without using sklearn ](https://github.com/PravinRedoc/Research/blob/main/SGD_imlplementation_Logloss.ipynb) 
* Implementaion of Stochastic Gradient Descent Classifier that optimizes logloss
* The model uses L2 regularization for generalizing model to avoid overfit/underfit
* The coeffients of the model is compared to Sklearn SGD classifer to prove the implementation, difference in the order of 10^-3

# [Research: Application of Bootstrap samples in Random forest](https://github.com/PravinRedoc/Research/blob/main/Bootstrap_RandomForest.ipynb) 
* Radomly create samples of data from the Boston housing dataset
* Sampled multiple features randomly as part Bagging.
* Computed the OOB score and CI and MSE to evaluate the aggregated model.

# [Research: Analysing the Behavior of Linear Models and Implementation of Decision function in Linear SVM classifier ](https://github.com/PravinRedoc/Research/tree/main/Linear%20models) 
* Implementaion of Decision function of SVM from scratch
* Compare decision function from sklearn and  compare the manual implementation.
* Compare various regularization parameters and its impact on model performance
* Analyse the impact of outlier with changing regularization parameters in linear models


# [Case Study: In-vehicle coupon recommendation - Predicting whether a person will accept the coupon recommended to him in different driving scenarios](https://github.com/PravinRedoc/Predictive-Analytics/blob/main/In_coupon_recommendation.ipynb) 
* The [dataset](https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation) was obtained from UCI machine learning repository.
* The data was pre-processed to conduct descriptive analytics and thorough analysis of the factors affecting the decision of the driver to accept the coupon was performed
* Features were studied and most important features were factored in.
* As part of predictive analytics, a range of machine learning models from Bayesian, Linear and Tree based Ensemble models were tuned using Cross validation and built (python sklearn)
* Feature Importance was analysed to understand the most valuable features in predicting the outcome.

# [Case Study: Personalized Cancer Diagnosis - Classifying Clinically Actionable Genetic Mutations](https://github.com/PravinRedoc/Predictive-Analytics/blob/main/CancerDiagnosisPersonalized.ipynb) 
* The [dataset](https://www.kaggle.com/c/msk-redefining-cancer-treatment/) was obtained from Kaggle in association with MSKCC
* The data was preprocessed and made model read to perform classification of the given genetic variations/mutations based on evidence from text-based clinical literature.
* NLP feature vectorization was performed using the nltk library for text features(Gene text)
* Text features were encoded using one-hot and response coding and the model giving best hyper parameters were chosen
* Various machine learning models were used such as Linear models(LR using SGD logloss and linear SVMs), Random Forest Classifier etc.
* All the models were tuned using logloss as the evaluation metric.
* The models were further examined using Confusion matrix, Precision and Recall matrices.
* The best features were picked using various feature importance techniques and the validity of each feature(Gene text vectorized, Variation) across Training, test and CV was checked.
* The model with best performance was chosen(Tuning).

# [Case Study: Quora Question Pair Similarity - Predicting whether a question is duplicate of another question given a pair of questions](https://github.com/PravinRedoc/Quora_QPS) 
* The [dataset](https://www.kaggle.com/c/quora-question-pairs) was obtained from Kaggle published by Quora.
* The data cleaned using text preprocessing technique such as steming, lemmatazing etc.
* Text features(Questions) were vectorized using GloVe vectors, one-hot encoding.
* Multiple text features were further extracted/engineered using fuzzywuzzy and nltk libraries.
* The effectiveness features extracted was examined by visualizing using dimensionality reduction technique such as T-SNE.
* The data was loaded into an SQLite database for better accessibilty and convenience.
* A range of machine learning models from Bayesian, Linear and Tree based Ensemble models were tuned using Cross validation were built and evaluated against a random model using logloss as the evaluation metric.












