# Gradient Boosting Ordinal Regression
*ordinal-boost* is the Python implementation of a novel mathematical framework for adapting machine learning (ML) models to handle ordinal response variables. Our approach is inspired by functional gradient descent used in gradient boosting and can be viewed as an extension of coordinate descent, where an ML regression model is embedded within a standard ordinal regression framework (e.g., logit or probit). The training process involves alternating between refining the ML model to predict a latent variable and adjusting the threshold vector applied to this latent variable to produce the ordinal response.

The workhorse is the `BoostedOrdinal` class, which conforms to `scikit-learn` conventions, including implementation of `fit`, `predict` and `predict_proba` methods. This allows for easy wrapping of the learner in hyperparameter tuning facilities provided in scikit-learn such as `GridSearchCV`.

- (add link to tutorial notebook)
- (add link to arxiv paper)

