import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt 
import pylab
import time

#Data Points
p1 = np.array([1,7])
p2 = np.array([2,5])
p3 = np.array([3,4])
p4 = np.array([4,5])
p5 = np.array([5,4])
p6 = np.array([6,4])
p7 = np.array([7,3])
p8 = np.array([4,4])
p9 = np.array([5,3])
p10 = np.array([7,1])
p11 = np.array([8,1])
#New point 
p12 = np.array([2,4])
#Initializing beta vector
beta = np.array([0,0])

#Data Matrix
X = np.array([[1,p1[0]],[1,p2[0]],[1,p3[0]],[1,p4[0]],[1,p5[0]],[1,p6[0]],[1,p7[0]],[1,p8[0]],[1,p9[0]],[1,p10[0]],[1,p11[0]]])
Y = np.array([[p1[1]],[p2[1]],[p3[1]],[p4[1]],[p5[1]],[p6[1]],[p7[1]],[p8[1]],[p9[1]],[p10[1]],[p11[1]]])

XT = np.transpose(X)
beta = np.matmul(np.matmul(inv(np.matmul(XT,X)),XT),Y)
temp = np.matmul(XT,X)															#XT*X
A = inv(temp)																	#(XT*X)^-1
B = np.matmul(XT,Y)																#XT*Y
Xnew = np.array([1,p12[0]]).reshape(1,2)
Ynew = np.array([p12[1]]).reshape(1,1)
temp1 = inv(temp + np.matmul(np.transpose(Xnew),Xnew))							#(XT*X + P11T*P11)-1
temp2 = B + np.matmul(np.transpose(Xnew),Ynew)									#(XT*Y + P11T*P11)-1
betaUpdate = np.matmul(temp1,temp2)

print('Old Slope and Intercept : ',beta)
print('New slope and Intercept : ',betaUpdate)

xnew = np.array([[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9]])
ynew = np.zeros((1, 10))
y = beta[1]*X + beta[0]
ynew = betaUpdate[1]*xnew +betaUpdate[0]

#Plot
plt.scatter(p1[0],p1[1])
plt.scatter(p2[0],p2[1])
plt.scatter(p3[0],p3[1])
plt.scatter(p4[0],p4[1])
plt.scatter(p5[0],p5[1])
plt.scatter(p6[0],p6[1])
plt.scatter(p7[0],p7[1])
plt.scatter(p8[0],p8[1])
plt.scatter(p9[0],p9[1])
plt.scatter(p10[0],p10[1])
plt.scatter(p11[0],p11[1])
plt.scatter(p12[0],p12[1])

plt.plot(X,y,'b-', label="Data Line")
plt.title('Online Regression')
plt.plot(xnew,ynew,'r-', label="Updated Data Line")
plt.legend()
plt.show()
