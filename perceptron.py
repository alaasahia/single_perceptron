import numpy as np

class SinglePerceptron:
    def __init__(self, input_size: int, seed=42):
        np.random.seed = seed #the seed parameter to ensure reproducible results
        self._weights = np.random.rand(input_size)
        self._bias = np.random.rand()
        self.learning_rate = None

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

    #the instances are our inputs 
    def train(self, epochs, instances, labels, learning_rate = 0.01):
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
            print('epoch loss:', epoch_loss/len(instances), 'accuracy', sum(epoch_accuracy)/len(instances))


