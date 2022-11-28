import numpy as np
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
 
 
#Reading data
df = pd.read_csv('fods_1.csv')
 
 
#Getting Necessary rows and columns
X = df.iloc[:,0:2]
X = np.array(X)
 
X1 = np.transpose(X)[0]
X2 = np.transpose(X)[1]
 
X1 = np.reshape(X1,(len(X1),1))
X2 = np.reshape(X2,(len(X2),1))
 
Y = np.array(df.iloc[:,2:3])
 
 
#Function to find cost
def computeCost(a,b):
    return np.sum(np.square(a*X1+b*X2-Y))/2
    
 
#Create the weights arrays
count = 500
w1 = np.linspace(-2,2,count)
w2 = np.linspace(-2,2,count)
w1_arr,w2_arr = np.meshgrid(w1,w2)
 
 
#Creating a cost matrix
cost = [[0 for j in range(count)] for i in range(count)]
ifPresent = [[0 for j in range(count)] for i in range(count)]
 
 

#creating a function to plot our error contour plot
def plotError(cost):
    figi = plt.figure(2)
    axi = plt.axes(projection='3d',proj_type='ortho')
    for i in range(count):
        for j in range(count):
            costElem = computeCost(w1[i],w2[j])
            cost[i][j] = costElem
    

 
    cost = np.array(cost)
    errorArr = np.reshape(cost,(count,count))
    
    fig, ax = plt.subplots(1,1)
    fig.set_figheight(6)
    fig.set_figwidth(6)
    for i in range(count):
        for j in range(count):
            costElem = computeCost(w1[i],w2[j])
            cost[i][j] = costElem
    
    cost = np.array(cost)
    errorArr = np.reshape(cost,(count,count))

    ax.contour(w1_arr, w2_arr, errorArr, 100, cmap=matplotlib.cm.winter.reversed())


    ax.set_xlabel('W1')
    ax.set_ylabel('W2')
    ax.set_aspect(1)
    #finding and plotting the minimal point
    xmin, ymin = np.unravel_index(np.argmin(errorArr), errorArr.shape)
    ax.plot(w1_arr[xmin,ymin], w2_arr[xmin,ymin], marker="o", ls="", color = 'r')
    txt= "("+str(w1_arr[xmin,ymin].round(3))+","+str(w2_arr[xmin,ymin].round(3))+")"
    ax.text(w1_arr[xmin,ymin], w2_arr[xmin,ymin],txt)

    plt.title(f'Error contour plot, with minimum total error as {errorArr.min()}')

    plt.savefig('./ErrorContourPlot.png')

#creating a function to plot our four constrained cases, the 
# constraint regions and error function contours and the point where the minima occurs

def plotConstraint(id, cost):
    if id == 1:
        q = 0.5
        eta = 1.4
    elif id == 2:
        q = 1
        eta = 0.1
    elif id == 3:
        q = 2
        eta = 0.035
    elif id == 4:
        q = 4
        eta = 0.052
    else:
        print("bruh")
    
    _min = 9999999999999
    iMin, jMin = (0,0)
 
    #Creating cost array, and finding the minimum values that satisfy the constraints
    for i in range(count):
        for j in range(count):
            costElem = computeCost(w1[i],w2[j])
            cost[i][j] = costElem
            if np.abs(w1[i])**q + np.abs(w2[j])**q - eta <=0:
                _min = min(_min,costElem)
                if (_min==costElem):
                    iMin, jMin = (i,j)
 
 
    cost = np.array(cost)
    errorArr = np.reshape(cost,(count,count))

        
    #plotting the constraints
    C = np.abs(w1_arr)**q + np.abs(w2_arr)**q - eta
    
    ax.contour(w1_arr,w2_arr,C,[0])
    plt.title(f'Contour Plot in 2D, with q = {q}, eta = {eta} and \n MSE as {computeCost(w1[iMin], w2[jMin])/len(X1)}')
    #plotting the tangential point 
    ax.plot(w1[iMin], w2[jMin], marker = 'o', markersize=5, color = 'r')
    text = "(" +str(w1[iMin].round(3))+","+str(w2[jMin].round(3))+")"
    ax.text(w1[iMin], w2[jMin],text)
    ax.set_aspect(1)
 
    return errorArr



#running and saving all five plots: the error contour plot, and the four 
# constrained region contour plots
for i in range(0,6):
    id = i
    if (id==0):
        plotError(cost) 
    elif (1<=id<=4):
        fig, ax = plt.subplots(1,1)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        errorArr = plotConstraint(id, cost)
    
        ax.contour(w1_arr, w2_arr, errorArr, 100, cmap=matplotlib.cm.winter.reversed())
        ax.set_xlabel('W1')
        ax.set_ylabel('W2')

        plt.show()
        # plt.savefig('./ContourPlot'+str(id)+'.png')
        plt.close()