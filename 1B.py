import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from mpl_toolkits import mplot3d
 
df = pd.read_csv('./fods_1.csv')
df = (df - df.mean())/df.std()
training_set_80 = df.sample(frac=0.8)
mf = np.array(training_set_80).max() - np.array(training_set_80).min()
test_Dataset_20 = df.drop(training_set_80.index)
 
X = training_set_80.iloc[:,0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
 
X1,X2 = X[:,1], X[:,2]
X1 = np.reshape(X1, (len(X1),1))
X2 = np.reshape(X2, (len(X2),1))
 
Y = np.array(training_set_80.iloc[:,2:3])
 
X_test = test_Dataset_20.iloc[:,0:2]
X_test = np.concatenate((np.ones([X_test.shape[0],1]),X_test),axis=1)
Y_test = np.array(test_Dataset_20.iloc[:,2:3])
 
X1_test,X2_test = X_test[:,1:2], X_test[:,2:3]
X1_test = np.reshape(X1_test, (len(X1_test),1))
X2_test = np.reshape(X2_test, (len(X2_test),1))
 
def makePoly(X1,X2,n):
  X= np.array([]).reshape((len(X1),0))
  for i in range(n, -1, -1):
    n1 = n-i+1
    for j in range(n1):
      X = np.c_[X,(X1**(n1-j-1))*(X2**j) ]
  return X
 
def normalize(matrix):
    matrix = np.delete(matrix, np.s_[0], 1)
    for i in range(len(matrix[1])):
        col = matrix[:,i]
        col = (col - col.mean())/(col.std())
        matrix[:,i] = col
    ones = np.ones((len(matrix),1))
    matrix = np.append(ones, matrix, 1)
    return matrix
 
 
def computeCost(X,Y,theta):
    square_error = np.power(((X @ theta.T)-Y),2)
    return np.sum(square_error)/(2*len(X))


def cost_L0_5(X,y,theta,lambda_):
    return  (((1/2)*np.sum(np.square((X@theta.T)-y)))+(lambda_*(np.sum(np.absolute(theta)**0.5))))
 
def cost_L1(X,y,theta,lambda_):
    return  (((1/2)*np.sum(np.square((X@theta.T)-y)))+(lambda_*(np.sum(np.absolute(theta)))))
 
def cost_L2(X,y,theta,lambda_):
    return  (((1/2)*np.sum(np.square((X@theta.T)-y)))+(lambda_*(np.sum(np.square(theta)))))
 
def cost_L4(X,y,theta,lambda_):
    return  (((1/2)*np.sum(np.square((X@theta.T)-y)))+(lambda_*(np.sum(theta**4))))


def gradientDescent(X,Y,max_iter=1000, alpha=0.01, threshold = 1e-6):
    costs = []
    currTheta = np.zeros((1,len(X[0])))
    prevCost = None
    for i in range(max_iter):
        currTheta = currTheta - (alpha/len(X)) * np.sum(X * (X @ currTheta.T - Y), axis=0)
        currCost = computeCost(X, Y, currTheta)
        if prevCost and abs(prevCost-currCost)<=threshold:
            break
        prevCost = currCost
        costs.append(currCost)

    return currTheta,costs
 
def gradient_descent_with_L2_regularisation(X,y,theta,learning_rate,iterations,lambda_):
    cost_history=np.zeros(iterations)
    
    for i in range(iterations):
        
        theta = theta - (learning_rate) * ((1/len(X))*(np.sum(X * ((X@theta.T) - y), axis=0)) +  lambda_*theta)
        cost_history[i] = cost_L2(X, y, theta,lambda_)
        
    return theta,cost_history
 
def gradient_descent_with_L4_regularisation(X,y,theta,learning_rate,iterations,lambda_):
    cost_history=np.zeros(iterations)
    
    for i in range(iterations):
        
        theta = theta - (learning_rate) * ((1/len(X))*(np.sum(X * ((X@theta.T) - y), axis=0)) +  2*lambda_*(theta)**3)
        cost_history[i] = cost_L4(X, y, theta,lambda_)
        
    return theta,cost_history
 
def gradient_descent_with_L1_regularisation(X,y,theta,learning_rate,iterations,lambda_):
    cost_history=np.zeros(iterations)
    
    for i in range(iterations):
        temp=np.array(theta)
        
        for j in range(theta.shape[1]):
        
            if(theta[0][j]>=0):
                temp[0][j]=1
            else:
                temp[0][j]=-1
        
 
        theta = theta - (learning_rate) * ((1/len(X))*(np.sum(X * ((X@theta.T) - y), axis=0)) +  lambda_*temp)
        
        cost_history[i] = cost_L1(X, y, theta,lambda_)
        
    return theta,cost_history
 
def gradient_descent_with_L0_5_regularisation(X,y,theta,learning_rate,iterations,lambda_):
    cost_history=np.zeros(iterations)
    
    for i in range(iterations):
        temp=np.array(theta)
        
        for j in range(theta.shape[1]):
        
            if(theta[0][j]>=0):
                temp[0][j]=temp[0][j]**(-0.5)
            else:
                temp[0][j]=-(abs(temp[0][j])**(-0.5))
        
 
        theta = theta - (learning_rate) * ((1/len(X))*(np.sum(X * ((X@theta.T) - y), axis=0)) +  ((0.25)*lambda_)*temp)
        
        cost_history[i] = cost_L0_5(X, y, theta,lambda_)
        
    return theta,cost_history
 
def _3dPlot(X1,X2,Y,Ypredicted,degree, X1_test, X2_test, Y_test):
    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(X1_test, X2_test, Y_test, c = 'b', s = 5)
    ax.scatter(X1, X2, Y, c = 'r', s = 5)
 
    
    surface = ax.plot_trisurf(X1.flatten(),X2.flatten(),Ypredicted.flatten(),cmap=matplotlib.cm.summer, linewidth=0,antialiased=False, alpha = 0.7)
 
    ax.set_xlabel('MLOGP', labelpad=20)
    ax.set_ylabel('GATS1i', labelpad=20)
    ax.set_zlabel('LC50', labelpad=20)
    ax.set_title("Predicted Polynomial for regression model of degree "+str(degree))
    fig.colorbar(surface, shrink=0.5, aspect=5)
    pl.show()
    pl.close(fig)
 
def sgd(x, y, learn_rate=0.001, batch_size=20, n_iter=1000,threshold=1e-06):
    n_obs = x.shape[0]
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]
 
    rng = np.random.default_rng()
    vector = np.zeros((1,x.shape[1]))
    learn_rate = np.array(learn_rate, dtype="float64")
    batch_size = int(batch_size)
    threshold = np.array(threshold, dtype="float64")
    for _ in range(n_iter):
        rng.shuffle(xy)
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
            grad = np.array(np.mean(x_batch * (x_batch @ vector.T - y_batch), axis=0), "float64")
            diff = -learn_rate * grad
            if np.all(np.abs(diff) <= threshold):
                break
            vector += diff
 
    return vector if vector.shape else vector.item()
 
 
errors = [0]*10
SGDerrors = [0]*10

trainerrors = [0]*10

 
for i in range(2,11):
    currentX = makePoly(X1, X2, i-1)
    currentX = normalize(currentX)
    currentXTest = makePoly(X1_test, X2_test, i-1)
    obtained_theta,costs = gradientDescent(currentX,Y)
    theta_sgd = sgd(currentX,Y, learn_rate=0.0001, batch_size=20, n_iter=1500)
    error = computeCost(currentXTest,Y_test,obtained_theta)
    SGDerrors[i-1] = computeCost(currentXTest,Y_test,theta_sgd)
    prediction = currentX @ obtained_theta.T
    _3dPlot(X1,X2,Y,prediction,i-1, X1_test, X2_test, Y_test)
    pl.plot([i for i in range(1,len(costs)+1)],costs,label='training error')
    pl.xlabel("Iterations")
    pl.ylabel("Error")
    pl.title(("Training Error for regression model of degree "+str(i-1)))
    pl.legend( loc='upper left')
    pl.savefig('./Training_Errors/degree'+str(i-1)+'.png')
    pl.close()
    errors[i-1] = error
    trainerrors[i-1] = costs[-1]
 
for i in range(1,10):
    print("Testing Error for degree", i ,"is:",errors[i])
print("Best Degree is: ", errors.index(min(errors[1:])), ", with error of: ", min(errors[1:]))
print()
for i in range(1,10):
    print("SGD testing Error for degree", i ,"is:",SGDerrors[i])
print("Best Degree SGD is: ", SGDerrors.index(min(SGDerrors[1:])), ", with error of: ", min(SGDerrors[1:]))
 

#Regularisation
 
X_multi = makePoly(X1, X2, 9)
X_multi = normalize(X_multi)
 
X_test_multi = makePoly(X1_test, X2_test, 9)
X_test_multi = normalize(X_test_multi)
 

points = 25
L1e = [0]*points
L1e_train = [0]*points
LambdaArr = []
for i in range(1, points+1):
    lamda = 0.025*i
    LambdaArr.append(lamda)
    L1theta = np.ones((1, X_multi.shape[1]))
    L1theta, hel = gradient_descent_with_L1_regularisation(X_multi, Y, L1theta, 0.01, 1000, lamda)
    L1e[i-1] = np.sum(((X_test_multi@L1theta.T-Y_test)**2))/(2*len(X_test_multi))

pl.plot(LambdaArr, L1e, label = "Testing Error")
pl.xlabel("Lambda")
pl.ylabel("Testing Error")
pl.title(("Lambda vs Testing Error for L1 Regularization"))
pl.legend( loc='upper left')
pl.savefig('./Training_Errors/LambdaVsTestingErrorL1.png')
pl.close()

print("\nBest Value of Lambda for L1 found to be: ", ((L1e.index(min(L1e)))+1)*0.025)
print("L1: ", min(L1e), "\n")




points = 50
L2e = [0]*points
L2e_train = [0]*points
LambdaArr = []
for i in range(1, points+1):
    lamda = 0.08*i
    LambdaArr.append(lamda)
    L1theta = np.ones((1, X_multi.shape[1]))
    L1theta, hel = gradient_descent_with_L2_regularisation(X_multi, Y, L1theta, 0.01, 1000, lamda)
    L2e[i-1] = np.sum(((X_test_multi@L1theta.T-Y_test)**2))/(2*len(X_test_multi))

pl.plot(LambdaArr, L2e, label = "Testing Error")
pl.xlabel("Lambda")
pl.ylabel("Testing Error")
pl.title(("Lambda vs Testing Error for L2 Regularization"))
pl.legend( loc='upper left')
pl.savefig('./Training_Errors/LambdaVsTestingErrorL2.png')
pl.close()

print("Best Value of Lambda for L2 found to be: ", ((L2e.index(min(L2e)))+1)*0.08)
print("L2: ", min(L2e), "\n")



points = 25
L0_5e = [0]*points
L0_5e_train = [0]*points
LambdaArr = []
for i in range(1, points+1):
    lamda = 0.024*i
    LambdaArr.append(lamda)
    L1theta = np.ones((1, X_multi.shape[1]))
    L1theta, hel = gradient_descent_with_L0_5_regularisation(X_multi, Y, L1theta, 0.01, 1000, lamda)
    L0_5e[i-1] = np.sum(((X_test_multi@L1theta.T-Y_test)**2))/(2*len(X_test_multi))

pl.plot(LambdaArr, L0_5e, label = "Testing Error")
pl.xlabel("Lambda")
pl.ylabel("Testing Error")
pl.title(("Lambda vs Testing Error for L0.5 Regularization"))
pl.legend( loc='upper left')
pl.savefig('./Training_Errors/LambdaVsTestingErrorL0_5.png')
pl.close()

print("Best Value of Lambda for L0.5 found to be: ", ((L0_5e.index(min(L0_5e)))+1)*0.024)
print("L0.5: ", min(L0_5e), "\n")




points = 30
L4e = [0]*points
L4e_train = [0]*points
LambdaArr = []
for i in range(1, points+1):
    lamda = i
    LambdaArr.append(lamda)
    L1theta = np.ones((1, X_multi.shape[1]))
    L1theta, hel = gradient_descent_with_L4_regularisation(X_multi, Y, L1theta, 0.01, 1000, lamda)
    L4e[i-1] = np.sum(((X_test_multi@L1theta.T-Y_test)**2))/(2*len(X_test_multi))

pl.plot(LambdaArr, L4e, label = "Testing Error")
pl.xlabel("Lambda")
pl.ylabel("Testing Error")
pl.title(("Lambda vs Testing Error for L4 Regularization"))
pl.legend( loc='upper left')
pl.savefig('./Training_Errors/LambdaVsTestingErrorL4.png')
pl.close()

print("Best Value of Lambda for L4 found to be: ", ((L4e.index(min(L4e)))+1)*1)
print("L4: ", min(L4e), "\n")