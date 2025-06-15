from sklearn.linear_model import Lasso

def lasso(X, y, alpha=0.1):
    """
    Use sklearn's Lasso implementation to solve the Lasso problem.
    """
    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_model.fit(X, y)  
    u = lasso_model.coef_  
    return u 
