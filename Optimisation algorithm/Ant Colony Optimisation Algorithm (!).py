import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical


class AntColonyOptimization:
    def __init__(self, num_vertices, graph, decay=0.1, generations=10, ants=3):
        self.num_vertices = num_vertices
        self.graph = graph
        self.decay = decay
        self.generations = generations
        self.ants = ants
        self.pheromones = np.ones((num_vertices, num_vertices))

    def choose_vertex(self, curr_vertex):
        graph = 1 / self.graph
        edges = graph[curr_vertex]
        numer = self.pheromones[curr_vertex].reshape((1, -1)) * edges.reshape((1, -1))


        denom = np.dot(self.pheromones[curr_vertex], edges)
        probabilities = numer / denom
        roulette_wheel = np.cumsum(probabilities)
        roulette_ball = np.random.random()
        next_vertex = np.where(roulette_wheel > roulette_ball)[0][0]
        return next_vertex

    def traverse(self, start, end):
        path = [start]
        cost = 0
        curr = start
        prev = start
        while curr != end:
            next_vertex = self.choose_vertex(curr)
            while next_vertex == prev:
                next_vertex = self.choose_vertex(curr)
            cost += self.graph[curr][next_vertex]
            path += [next_vertex]
            prev = curr
            curr = next_vertex
        return path, cost

    def release_ants(self, start, end):
        paths = []
        costs = []
        for i in range(self.ants):
            p, c = self.traverse(start, end)
            paths += [p]
            costs += [c]
        return paths, costs

    def update_pheromones(self, paths, costs):
        costs = np.array(costs)
        costs = 1 / costs
        self.pheromones = (1 - self.decay) * self.pheromones
        for p in range(len(paths)):
            path = paths[p]
            for v in range(len(path) - 1):
                self.pheromones[path[v]][path[v + 1]] += costs[p]
        return self.pheromones


df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

# Select features and target
X = df[['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard']].values
y = df['Personal Loan'].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


n_ants = 50
pheromone = np.ones((X_train.shape[1], n_ants))


heuristic = 1 / np.abs(np.random.randn(X_train.shape[1], n_ants))
evaporation_rate = 0.1
n_iterations = 100


n_inputs = X_train.shape[1]
n_outputs = n_ants
n_hidden_layers = 3
n_hidden_nodes = 32
activation = 'relu'
learning_rate = 0.001
model = Sequential()
model.add(Dense(n_hidden_nodes, activation=activation, input_shape=(n_inputs,)))
for i in range(n_hidden_layers-1):
    model.add(Dense(n_hidden_nodes, activation=activation))
model.add(Dense(n_outputs, activation='softmax'))
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

output = model(X_train)
aco = AntColonyOptimization(num_vertices=X_train.shape[1], graph=heuristic, decay=evaporation_rate, generations=n_iterations, ants=n_ants)

# Train the model with ACO-based feature selection
for i in range(n_iterations):
    # Select features using ACO
    paths, costs = aco.release_ants(start=0, end=n_ants-1)
    pheromone = aco.update_pheromones(paths, costs)

    # Convert paths to binary masks for feature selection
    masks = np.zeros((n_ants, n_inputs))
    for i, path in enumerate(paths):
        masks[i, path] = 1

    # Train the model with the selected features
    for mask in masks:
        # Split data into train and validation sets
        X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(X_train * mask, y_train, test_size=0.2, random_state=42)

        # One-hot encode the targets
        y_train_sel = to_categorical(y_train_sel)
        y_val_sel = to_categorical(y_val_sel)

        # Train the model
        history = model.fit(X_train_sel, y_train_sel, epochs=50, batch_size=32, verbose=0, validation_data=(X_val_sel, y_val_sel))

        # Evaluate the model
        score = model.evaluate(X_test * mask, to_categorical(y_test), verbose=0)

        # Print the results
        print('Features:', np.where(mask == 1)[0])
        print('Accuracy:', score[1])
        print('Loss:', score[0])
