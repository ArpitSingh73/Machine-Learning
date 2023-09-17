import numpy as np



class SGRegressor:
    
    def __init__(self,learning_rate=0.01,epochs=100):
        
        self.coef = None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
    
        self.intercept = 0
        self.coef = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0,X_train.shape[0])
                
                y_hat = np.dot(X_train[idx],self.coef) + self.intercept                 
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept = self.intercept - (self.lr * intercept_der)
                
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef = self.coef - (self.lr * coef_der)
        
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef) + self.intercep 