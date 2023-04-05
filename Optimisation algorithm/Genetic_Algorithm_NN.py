import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# GeneticAlgorithm class definition
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        
    def run(self, fitness_fn, gene_pool):
        # Initialising the initial population
        population = self.initialize_population(gene_pool)
        
        # Iterate generations
        for i in range(self.num_generations):
            # Evaluate the fitness of the population
            fitness_scores = [fitness_fn(individual) for individual in population]
            
            # Select the parents for the next generation (use fitness score to chose)
            parents = self.select_parents(population, fitness_scores)
            
            # Next generation
            next_generation = self.create_next_generation(parents, gene_pool)
            
            # Mutation
            mutated_generation = self.mutate_population(next_generation, gene_pool)
            
            # Update
            population = mutated_generation
        
        # Return the best individual from the final generation
        return max(population, key=fitness_fn)
    
    def initialize_population(self, gene_pool):
        population = []
        for i in range(self.population_size):
            individual = [np.random.choice(gene_pool[param]) for param in gene_pool]
            population.append(individual)
        return population
    
    def select_parents(self, population, fitness_scores):
        parents = []
        total_fitness = sum(fitness_scores)
        for i in range(len(population)):
            # Choose two parents based on roulette wheel selection
            parent1 = self.roulette_wheel_selection(population, fitness_scores, total_fitness)
            parent2 = self.roulette_wheel_selection(population, fitness_scores, total_fitness)
            parents.append((parent1, parent2))
        return parents
    
    def roulette_wheel_selection(self, population, fitness_scores, total_fitness):
        # Calculate the probabilities of each individual being selected
        probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]
        # Choose an individual based on the probabilities using roulette wheel selection
        return population[np.random.choice(len(population), p=probabilities)]
    
    def create_next_generation(self, parents, gene_pool):
        next_generation = []
        for parent1, parent2 in parents:
            # Perform single-point crossover
            crossover_point = np.random.randint(len(parent1))
            child = parent1[:crossover_point] + parent2[crossover_point:]
            next_generation.append(child)
        return next_generation
    
    def mutate_population(self, population, gene_pool):
        for i in range(len(population)):
            individual = population[i]
            # Mutate each gene with probability mutation_rate
            for j in range(len(individual)):
                if np.random.random() < self.mutation_rate:
                    individual[j] = np.random.choice(gene_pool[list(gene_pool.keys())[j]])
        return population

# Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, input_shape, output_shape, hidden_layer1_size, hidden_layer2_size, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size
        self.learning_rate = learning_rate
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_layer1_size, activation='relu', input_shape=(self.input_shape,)),
            tf.keras.layers.Dense(self.hidden_layer2_size,activation='relu'),
            tf.keras.layers.Dense(self.output_shape, activation='sigmoid')])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
        #return self.model
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        var = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                             epochs=epochs, batch_size=batch_size, verbose=0)
        return var
    
    def evaluate(self, X_val, y_val):
        loss, accuracy = self.model.evaluate(X_val, y_val)
        return loss, accuracy
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred


data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
X = data.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1)
y = data['Personal Loan']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

gene_pool = {
    'hidden_layer1_size': [32, 64, 128],
    'hidden_layer2_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1]
}

def fitness_fn(params):
    hidden_layer1_size = params[0]
    hidden_layer2_size = params[1]
    learning_rate = params[2]
    nn = NeuralNetwork(input_shape=X_train.shape[1], output_shape=1, 
                       hidden_layer1_size=hidden_layer1_size, hidden_layer2_size=hidden_layer2_size, 
                       learning_rate=learning_rate)
    var = nn.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=16)
    loss, accuracy = nn.evaluate(X_test, y_test)
    return accuracy




ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1, num_generations=5)
best_params=ga.run(fitness_fn, gene_pool)
nn = NeuralNetwork(input_shape=X_train.shape[1], output_shape=1,hidden_layer1_size=best_params[0],hidden_layer2_size= best_params[1],learning_rate=best_params[2])
var = nn.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=16)






loss, accuracy = nn.evaluate(X_test, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
y_pred = nn.model.predict(X_test)
y_pred_class = np.where(y_pred > 0.5, 1, 0)

cm = confusion_matrix(y_test, y_pred_class)
cr = classification_report(y_test, y_pred_class)
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', cr)



