import numpy as np

# Class for single feature-->
class GDRegressor:
    
    def __init__(self,learning_rate,epochs):
        self.m = 100
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X,y):
        # calcualte the b using GD
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m*X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m*X.ravel() - self.b)*X.ravel())
            
            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)
        print(self.m,self.b)
        
    def predict(self,X):
        return self.m * X + self.b





# Class for multiple features-->
class BatchGD:

    def __init__(self, lr=0.01, epoch=100):
        self.lr = lr
        self.epoch = epoch

    def fit(self, x_train, y_train):

        self.intercept = 0
        self.coeff = np.ones(x_train.shape[1])

        for i in range(self.epoch):
            y_hat = np.dot(x_train, self.coef_) + self.intercept

            intercept = -2 * np.mean(y_train - y_hat)
            self.intercept = self.intercept_ - (self.lr * intercept)
            
            coef_der = -2 * np.dot((y_train - y_hat),x_train)/x_train.shape[0]
            self.coeff = self.coef_ - (self.lr * coef_der)
        
        
    
    def predict(self,x_test):
        return np.dot(x_test,self.coeff) + self.intercept
