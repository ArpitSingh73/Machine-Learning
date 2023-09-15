import numpy as np


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
