## REFERENCES -RESEARCH PAPERS

### GENETIC ALGORITHM
###### [1] J.  Holland,  “Adaptation  in  natural  and  artificial  systems”, University of Michigan press, Ann Arbor, 1975
###### [2] A Study on Genetic Algorithm and its Applications" discusses the concept of genetic algorithm (GA) and its various applications ,Haldurai Lingaraj
###### [3] The Influence of Genetic Algorithms on Learning Possibilities of Artificial Neural Networks, Martin Kotyrba,University of Ostrava,Eva Volna,University of Ostrava,Hashim Habiballa,University of Ostrava,Josef Czyz
### ANT COLONY OPTIMISATION
###### [1] Ant Colony Optimization Algorithm,Nada M. A. Al Salami
###### [2]  M. Dorigo, M. Birattari, and T. Stitzle,“Ant Colony Optimization: Arificial Ants as a Computational Intelligence Technique, IEEE computational intelligence magazine, November,2006. 
### CULTURAL ALGORITHM 
###### [1] Reynolds,  R.G.,  “An  Overview  of  Cultural  Algorithms”,  Advances  in Evolutionary  Computation, McGraw Hill Press, 1999.
###### [2] Reynolds, R. G., "Introduction to Cultural Algorithms", in  Proceedings of the Third Annual Conference on Evolutionary Programming, Anthony V. Sebald and Lawrence J. Fogel, Editors, World Scientific Press, Singapore, 1994, pp.131-139
### PARTICLE SWARM OPTIMISATION
###### [1] . Trelea, I., The particle swarm optimization algorithm: convergence analysis and parameter selection,ELSEVIER Science, 2002 – Information Processing Letters 85, 317-325, 2003.
# Comparison of performances
Based on what I went through in the above mentioned research papers,
 Genetic Algorithms and Cultural algorithms can be computationally expensive, and GA's performance heavily depends on the choice of parameters such as population size, mutation rate, and crossover rate whereas CA's performance heavily depends on the choice of parameters such as belief space size, population size, and migration rate
 Ant Colony Optimisation and Particle Swarm Optimisation are computationally efficient however, ACO can be sensitive to the choice of parameter values such as pheromone evaporation rate, heuristic function, and number of ants and PSO can be sensitive to the swarm size, velocity limit, and inertia weight
 Some papers say that metaheuristic optimization algorithms, such as PSO and GA, may perform well in neural network training compared to other optimization algorithms
 The choice of optimization algorithm or technique for neural network training depends on the specific problem.
 
 I tried implementing the Genetic Algorithm with the fullest of my knowledge and was able to give 98% accuracy which has indeed optimised the accuracy of a simple whitebox neural network giving 96% accuracy.The ACO and PSO results were not convincing enough for me to give a comparison amongst all the 4 optimisation algorithms.My knowledge about cultural algorithm is very minimal so I wil implement the CA once I get a good idea of how the algorithm optimises the neural network.
