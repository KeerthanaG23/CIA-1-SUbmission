import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score

df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
X = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1)
y = df['Personal Loan']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros((1, self.output_dim))
        
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = np.round(1 / (1 + np.exp(-self.z2)))
        
        return self.y_hat
    
    def loss(self, X, y):
        # Calculate the loss
        y_hat = self.forward(X)
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        
        return loss
class PSO:
    def __init__(self, n_particles, n_iter, c1, c2, w):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        
        # Initialize the particles
        self.particles = []
        for i in range(self.n_particles):
            nn = NeuralNetwork(X_train.shape[1], 5, 1)
            self.particles.append(nn)
        
        # Initialize the best positions and fitnesses
        self.best_positions = []
        self.best_fitnesses = []
        for particle in self.particles:
            self.best_positions.append(particle)
            self.best_fitnesses.append(particle.loss(X_train, y_train))
            self.global_best_position = self.particles[np.argmin(self.best_fitnesses)]
            self.global_best_fitness = self.best_fitnesses[np.argmin(self.best_fitnesses)]
            
    
    # Initialize the velocities
        self.velocities = []
        for i in range(self.n_particles):
            v = NeuralNetwork(X_train.shape[1], 5, 1)
        #self.velocities.append(v).loss(X_train, y_train)
        self.global_best_position = self.particles[np.argmin(self.best_fitnesses)]
        self.global_best_fitness = self.best_fitnesses[np.argmin(self.best_fitnesses)]
          
          # Initialize the velocities
        self.velocities = []
        for i in range(self.n_particles):
            v = NeuralNetwork(X_train.shape[1], 5, 1)
            self.velocities.append(v)
          
    def update_velocity(self, particle):
          # Update the velocity
          r1 = np.random.rand(*self.velocities[0].W1.shape)
          r2 = np.random.rand(*self.velocities[0].W2.shape)
          
          self.velocities[particle].W1 = (self.w * self.velocities[particle].W1 
                                          + self.c1 * r1 * (self.best_positions[particle].W1 - self.particles[particle].W1)
                                          + self.c2 * r2 * (self.global_best_position.W1 - self.particles[particle].W1))
          
          r1 = np.random.rand(*self.velocities[0].b1.shape)
          r2 = np.random.rand(*self.velocities[0].b2.shape)
          
          self.velocities[particle].b1 = (self.w * self.velocities[particle].b1 
                                          + self.c1 * r1 * (self.best_positions[particle].b1 - self.particles[particle].b1)
                                          + self.c2 * r2 * (self.global_best_position.b1 - self.particles[particle].b1))
          
          r1 = np.random.rand(*self.velocities[0].W2.shape)
          r2 = np.random.rand(*self.velocities[0].b2.shape)
          
          self.velocities[particle].W2 = (self.w * self.velocities[particle].W2 
                                          + self.c1 * r1 * (self.best_positions[particle].W2 - self.particles[particle].W2)
                                          + self.c2 * r2 * (self.global_best_position.W2 - self.particles[particle].W2))
          
          self.velocities[particle].b2 = (self.w * self.velocities[particle].b2 
                                          + self.c1 * r1 * (self.best_positions[particle].b2 - self.particles[particle].b2)
                                          + self.c2 * r2 * (self.global_best_position.b2 - self.particles[particle].b2))
      
    def update_position(self, particle):
          # Update the position
          self.particles[particle].W1 += self.velocities[particle].W1
          self.particles[particle].b1 += self.velocities[particle].b1
          self.particles[particle].W2 += self.velocities[particle].W2
          self.particles[particle].b2 += self.velocities[particle].b2
      
    def evaluate_fitness(self, particle):
          # Evaluate the fitness
          fitness = self.particles[particle].loss(X_train, y_train)
          
          if fitness < self.best_fitnesses[particle]:
              self.best_positions[particle] = self.particles[particle]
              self.best_fitnesses[particle] = fitness
          
          if fitness < self.global_best_fitness:
              self.global_best_position = self.particles[particle]
              self.global_best_fitness = fitness
      
    def train(self):
          for i in range(self.n_iter):
              for particle in range(self.n_particles):
                  self.update_velocity(particle)
                  self.update_position(particle)
                  self.evaluate_fitness(particle)
                  
                  if i % 10 == 0:
                      print(f"Iteration {i}, Best fitness: {self.global_best_fitness}")
    
    print("Training completed.")
    
    def predict(self, X_test):
    # Make predictions using the trained neural network
        y_pred = self.global_best_position.predict(X_test)
        return y_pred

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the PSO class and train the model
pso = PSO(n_particles=10, n_iter=100, w=0.7, c1=1.4, c2=1.4)
pso.train(X_train, y_train)

# Make predictions on the testing set
y_pred = pso.predict(X_test)

# Evaluate the performance of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))


