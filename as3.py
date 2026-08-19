#    NAME: Jeremiah Monteverde, 5007922369, CS 422 - 1001,
#          Assignment 3
#    DESCRIPTION: An assignment focused on linear regression,
#                   and using both OLS and Gradient Descent in
#                   comparison.
#    INPUT: The student exam scores dataset from kaggle!
#    OUTPUT: Evaluation metrics for both test and training sets
#               via OLS method and Gradient Descent methods!

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Linear Regression Class - the OLS Method!
class OLS:
    # function dedicated to calculating the weight
    #   parameters include the model, inputs (e.g. X_train), 
    #   and targets (e.g. y_train)
    def weightCalc(self, input, targets):
        self.exa, self.fts = input.shape    # row by column

        # adding column of 1 to design matrix
        desTrix = np.ones((self.exa, 1))
        desTrix = np.hstack((desTrix, input))

        #print(desTrix.shape)
        #print(self.fts)

        TrDesTrix = desTrix.transpose()

        w = np.dot(desTrix, TrDesTrix)
        w = np.linalg.pinv(w)           # did psuedo instead of actual inv, fixed calc issues
        w = np.dot(TrDesTrix, w)
        w = np.dot(w, targets)

        #print(w.shape)

        return w
    # function dedicated to predicting class when given inputs + weights
    #   parameters include the model, inputs (e.g. X_train), 
    #   and the calculated weights
    def predict(self, input, weight):
        self.exa, self.fts = input.shape    # row by column

        # adding column of 1 to design matrix
        biasCol = np.ones((self.exa, 1))
        newInput = np.hstack((biasCol, input))
        #print(desTrix)

        prediction = np.dot(newInput, weight)
        return prediction
    
# a function dedicated to outputting calculated evaluation metrics
#   parameters include the actual labels + predicted labels
def evalMetrics(targets, predictions):
    # calculates MSE
    MSE = (targets - predictions) **2
    MSE = np.mean(MSE)
    print("  MSE = ", MSE)

    # calculates RMSE
    RMSE = np.sqrt(MSE)
    print("  RMSE = ", RMSE)

    # calculates MAE
    MAE = targets - predictions
    MAE = np.absolute(MAE)
    MAE = np.mean(MAE)
    print("  MAE = ", MAE)

    # calcualtes R^2, or the coefficient of determination
    baseline = np.mean(targets)
    bMSE = (targets - baseline) **2
    bMSE = np.mean(bMSE)
    r2 = 1 - (MSE / bMSE)
    print("  R^2 = ", r2)

# Linear Regression Class - the Gradient Descent Method!
class gradDes:
    # function for conducting gradient descent until convergence
    #   parameters include the model, inputs (e.g. X_train), 
    #   and targets (e.g. y_train)
    def iterations(self, inputs, targets):
        # starting with learning rate of 0.1
        lRate = 0.1

        self.exa, self.fts = inputs.shape    # row by column

        # adding column of 1 to design matrix
        biasInput = np.ones((self.exa, 1))
        biasInput = np.hstack((biasInput, inputs))

        # creating random weight array
        w = np.random.randn(self.fts + 1)
        w2 = w

        # enables use of AdaGrad to continuously update learning rate
        #sumOfGrad = np.zeros(self.fts + 1)  # populate w/ future squared gradients

        itCount = 0 # iteration counter for report
        convergence = 0     # variable for whether convergence was reached

        while convergence == 0:  # iteration cap is 1000
            itCount += 1    # increments itCount

            gradient = np.dot(biasInput, w)
            gradient -= targets
            gradient = np.dot(biasInput.transpose(), gradient)
            gradient /= self.exa
            
            #sumOfGrad += gradient **2

            #lRate /= np.sqrt(sumOfGrad) + 1e-8 
            
            w2 -= lRate * gradient

            if np.linalg.norm(gradient) < 1e-5:
                convergence = 1

        print("# of Iterations till Convergence: ", itCount)
        return w2
    
    # function dedicated to predicting class when given inputs + weights
    #   parameters include the model, inputs (e.g. X_train), 
    #   and the calculated weights
    def predict(self, input, weight):
        self.exa, self.fts = input.shape    # row by column

        # adding column of 1 to design matrix
        biasCol = np.ones((self.exa, 1))
        newInput = np.hstack((biasCol, input))
        #print(desTrix)

        prediction = np.dot(newInput, weight)
        return prediction

def main() :
    # grabbing the dataset to store into a panda
    baseData = pd.read_csv("student_exam_scores.csv")

    # print("Panda Contents:\n ", baseData)

    # setting training data (X), and its labels (y)
    X = baseData.iloc[:, 1:-1]  # excludes first column (id) and last column
    y = baseData.iloc[:, -1]    # only contains last column

    #print("Panda Contents:\n ", y)

    # split the data into training and testing sets!
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2, random_state=42)
    
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    
    # grabbing means and standard deviation of each feature
    colMeans = X_train.mean(axis=0)     
    colStd = X_train.std(axis=0)

    # standardization time
    stdX_train = (X_train - colMeans) / colStd
    stdX_test = (X_test - colMeans) / colStd

    model = OLS()
    w = model.weightCalc(stdX_train, y_train)
    predY_train = model.predict(stdX_train, w)
    #print(y_train.shape)
    #print(X_train.shape)

    print("Training Set's Evaluation Metrics via OLS Method")
    evalMetrics(y_train, predY_train)
    print("  Solution w = ", w)

    w = model.weightCalc(stdX_test, y_test)
    predY_test = model.predict(stdX_test, w)

    print("\nTest Set's Evaluation Metrics via OLS Method")
    evalMetrics(y_test, predY_test)
    print("  Solution w = ", w)

    model2 = gradDes()
    print("\nTraining Set:")
    w2 = model2.iterations(stdX_train, y_train)
    predY_train2 = model2.predict(stdX_train, w2)

    print("\nTraining Set's Evaluation Metrics via Gradient Descent Method")
    evalMetrics(y_train, predY_train2)
    print("  Solution w = ", w2)

    print("\nTest Set:")
    w2 = model2.iterations(stdX_test, y_test)
    predY_test2 = model2.predict(stdX_test, w2)

    print("\nTest Set's Evaluation Metrics via Gradient Descent Method")
    evalMetrics(y_test, predY_test2)
    print("  Solution w = ", w2)


if __name__ == "__main__" :
    main()