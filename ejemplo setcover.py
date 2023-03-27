import numpy as np
import random

filename = "rail507.txt"
print("Leyendo archivo...")

# Crear la lista de conjuntos
sets = []
with open(filename) as f:
    for line in f:
        s = {int(x) for x in line.strip().split()}
        sets.append(s)
        print("Conjunto leído: ", list(s))

# Crear el universo de elementos
universeSize = len(sets)
universe = set(range(universeSize))

# Crear un array binario de tamaño igual al número de conjuntos
solution = [False] * len(sets)

# Definir los parámetros del algoritmo genético
POP_SIZE = 1000 # Tamaño de la población
GENERATIONS = 100 # Número de generaciones
MUTATION_PROB = 0.1 # Probabilidad de mutación
FITNESS_THRESHOLD = len(universe) # Umbral de aptitud para la solución

# Definir la función de aptitud
def fitness(solution):
    """
    Calcula la aptitud de una solución dada.
    La aptitud se define como el número de elementos del universo
    que están cubiertos por la solución.
    """
    covered = set()
    for i, s in enumerate(sets):
        if solution[i]:
            covered |= s
    return len(covered)

# Definir los operadores genéticos
def crossover(parent1, parent2):
    """
    Cruza dos padres para crear un nuevo hijo.
    """
    child = [False] * len(parent1)
    midpoint = random.randint(0, len(parent1)-1)
    child[:midpoint] = parent1[:midpoint]
    child[midpoint:] = parent2[midpoint:]
    return child

def mutate(solution):
    """
    Mutar una solución.
    En este caso, la mutación consiste en invertir el valor de un gen.
    """
    mutated = solution.copy()
    index = random.randint(0, len(mutated)-1)
    mutated[index] = not mutated[index]
    return mutated

# Generar una población inicial aleatoria
population = [np.random.randint(2, size=len(solution)).tolist() for _ in range(POP_SIZE)]

# Iterar hasta que se encuentre una solución satisfactoria o se alcance el número máximo de generaciones
for generation in range(GENERATIONS):
    # Calcular la aptitud de cada individuo de la población
    fitness_scores = [fitness(individual) for individual in population]

    # Seleccionar a los padres para la próxima generación mediante torneo binario
    parents = []
    for _ in range(POP_SIZE):
        candidate1, candidate2 = random.sample(range(POP_SIZE), k=2)
        if fitness_scores[candidate1] >= fitness_scores[candidate2]:
            parents.append(population[candidate1])
        else:
            parents.append(population[candidate2])

    # Cruzar a los padres para crear una nueva población
    new_population = []
    for i in range(0, POP_SIZE, 2):
        child1 = crossover(parents[i], parents[i+1])
        child2 = crossover(parents[i+1], parents[i])
        new_population.extend([child1, child2])

    # Mutar a la nueva población
    for i in range(POP_SIZE):
        if random.random() < MUTATION_PROB:
            mutated_child = mutate(new_population[i])
            new_population[i] = mutated_child
            
    # Reemplazar la población anterior con la nueva población generada
    population = new_population

    # Encontrar la solución más apta de la población actual
    best_individual = population[np.argmax(fitness_scores)]

    # Si se encuentra una solución perfecta, detener la iteración
    if fitness(best_individual) == FITNESS_THRESHOLD:
        print("Solución encontrada en la generación ", generation)
    break
        
# Mostrar la solución encontrada
print("Mejor solución encontrada en la generación {}: {}".format(generation, best_individual))

# Obtener los conjuntos seleccionados
selected_sets = [sets[i] for i, value in enumerate(best_individual) if value]

# Encontrar el conjunto mínimo
min_set = set.union(*selected_sets)

# Mostrar el conjunto mínimo
print("Conjunto mínimo en la generación {}: {}".format(generation, list(min_set)))
   