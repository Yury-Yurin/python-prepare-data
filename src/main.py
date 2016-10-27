import numpy as np
from sklearn.decomposition import PCA

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-0.1 * x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i]
                                                       + 1)) - 1) * 0.25)
        self.weights.append((2 * np.random.random((layers[i] + 1, layers[i +
                                                                         1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


dataInFile = open('/home/yury/BSTU/lab3/correctedNew')
dataOutFile = open('/home/yury/BSTU/lab3/attacks')
mainComponents = open('/home/yury/BSTU/lab3/mainComponents','w+')
main = open('/home/yury/BSTU/lab3/main','w+')
lines = dataInFile.readlines()
lines1 = dataOutFile.readlines()
trainingData = list()
outData = list()
for i in range(0,311029):
    t = float(lines1[i])
    outData.append(t)
pca = PCA(n_components=20)
for i in range(0,311029):
    params = lines[i].split(',')
    t = list()
    r = float(lines1[i])
    for j in params:
        s = float(j)
        t.append(s)
    trainingData.append(t)
print(trainingData.__len__())
print(outData.__len__())
fit = pca.fit(trainingData)
features = pca.transform(trainingData)
for i in range(0,outData.__len__()):
    main.write(str(outData[i]) + '\n')
    for j in range(0,20):
        if(j!=19):
            mainComponents.write(str(features[i][j]) + ':')
        else:
            mainComponents.write(str(features[i][j]) + '\n')