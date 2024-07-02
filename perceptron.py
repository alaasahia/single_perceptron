import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
class SinglePerceptron:
    def __init__(self, input_size: int, seed=42):
        np.random.seed = seed #the seed parameter to ensure reproducible results
        self._weights = np.random.rand(input_size)
        self._bias = np.random.rand()
        self.learning_rate = None
        self.reduced_data = None

    def forward(self, x):
        out = x @ self._weights + self._bias
        return out

    def backward(self, error: float, x):
        self._weights += self.learning_rate * error * x
        self._bias += error * self.learning_rate

    def step(self, out, treshold=0.5):
        return 1 if out >= treshold else 0

    def loss(self, out, target):
        return target - out
    #visualize the data according to its labels
    def visualize_data(self, data, labels):
        #reduce the dimensions of the data if it has a shape or a size bigger then 2
        #in this case we used the PCA algorithm
        if data.shape[0] > 2:
            reducer = PCA(n_components=2)
            self.reduced_data = reducer.fit_transform(data)
        else:
            self.reduced_data = data
        #mark the data point with a color based on its label
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = labels == label
            plt.scatter(self.reduced_data[indices, 0], self.reduced_data[indices, 1], label=f'class: {label}')
        plt.title(f'Data Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()

    #the instances are our inputs 
    def train(self, epochs, instances, labels, learning_rate = 0.000001):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = []
            for i, inst in enumerate(instances):
                out = self.forward(inst)
                loss = self.loss(out, labels[i])
                epoch_loss += abs(loss)
                step_out = self.step(out)
                epoch_accuracy.append(1 if step_out == labels[i] else 0)
                self.backward(loss, inst)
            print(f'epoch {epoch+1}:','loss:', epoch_loss/len(instances), 'accuracy', sum(epoch_accuracy)/len(instances))
