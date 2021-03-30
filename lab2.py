import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

#kernel function
'''def kernel(x, y):
    return numpy.dot(x,y)'''

#polynomial kernels
def kernel(x, y):
    rtn = (numpy.dot(x, y) + 1) ** 2
    return rtn

#RBF kernels
'''def kernel(x, y):
    sigma = 10
    mod = numpy.dot((x-y), (x-y))
    return math.exp(-1/(2*sigma)*mod)'''

#Implement the function objective
numpy.random.seed(100)
classA = numpy.concatenate((numpy.random.randn(10,2)*0.5+[2,2], numpy.random.randn(10,2)*0.2+[-1.5,0.5]))
classB = numpy.random.randn(20,2)*0.5+[0.0,-0.5]
#classA = numpy.concatenate((numpy.random.randn(10,2)*0.2+[1.5,0.5], numpy.random.randn(10,2)*0.2+[-1.5,0.5]))
#classB = numpy.random.randn(20,2)*0.2+[0.0,-0.5]
inputs = numpy.concatenate((classA,classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0])))
N = inputs.shape[0]#Numberofrows(samples)
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

x = inputs
t = targets
P = [ [ 0 for i in range(N) ] for j in range(N) ]

for i in range(len(x)):
    for j in range(len(x)):
        K = kernel(x[i], x[j])
        P[i][j] = t[i]*t[j]*K


def objective(alpha):
    alpha_P = numpy.dot(alpha, P)
    alpha_P_alpha_slash = numpy.dot(alpha_P, numpy.transpose(alpha))
    return 1/2*alpha_P_alpha_slash-sum(alpha)


#Implement the function zerofun
def zerofun(alpha):
    return numpy.dot(alpha, t)


#Call minimize
start = numpy.zeros(N)
C = 20
#B = [(0, C) for b in range(N)]
B = [(0, None) for b in range(N)]
XC = {'type': 'eq', 'fun': zerofun}
ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']

#Extract the non-zero alpha values
alpha_greater_than_0 = list()
support_vectors = list()
sv_target = list()
for i in range(len(alpha)):
    if alpha[i] > 1e-5:
        alpha_greater_than_0.append(alpha[i])
        support_vectors.append(x[i])
        sv_target.append(t[i])
print(alpha_greater_than_0)
print(support_vectors)
print(sv_target)

#Calculate the b value using equation (7)
if (len(alpha_greater_than_0) == 0):
    print("cannot find support vectors")
    sys.exit()
s = support_vectors[0]
t_s = sv_target[0]
b_temp = 0
for i in range(N):
    b_temp = b_temp + alpha[i]*t[i]*kernel(s, x[i])
b = b_temp - t_s


#Implement the indicator function
def indicator(x1, y1):
    ind_temp = 0
    for i in range(N):
        ind_temp = ind_temp + alpha[i] * t[i] * kernel([x1, y1], x[i])
    return ind_temp-b


#plot
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')#Forcesamescaleonbothaxes

#plot support vectors
plt.plot([p[0] for p in support_vectors], [p[1] for p in support_vectors], 'g+')


xgrid=numpy.linspace(-5,5)
ygrid=numpy.linspace(-4,4)
grid=numpy.array([[indicator(x,y) for x in xgrid] for y in ygrid])
plt.contour(xgrid,ygrid,grid, (-1.0,0.0,1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

plt.savefig('svmplot.pdf')#Saveacopyinafile
#plt.title("C="+str(C))
plt.show()#Showtheplotonthescreen