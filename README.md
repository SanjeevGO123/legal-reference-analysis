# Algorithm Performance Comparison (MRR and MAP for Varying Relevance Thresholds)

| **Relevance Threshold** | **ACO (MRR & MAP)** | **PSO (MRR & MAP)** | **Firefly (MRR & MAP)** | **Genetic (MRR & MAP)** | **BAT (MRR & MAP)** | **ABC (MRR & MAP)** | **Cuckoo Search (MRR & MAP)** | **Flower Pollination (MRR & MAP)** |
|-------------------------|---------------------|---------------------|-------------------------|-------------------------|---------------------|---------------------|-----------------------------|----------------------------------|
| **0.05 to 0.09**        | 1 / 1               | 1 / 1               | 1 / 1                   | 1 / 1                   | 1 / 1               | 1 / 1               | 1 / 1                       | 1 / 1                            |
| **0.09 to 0.159**       | 0 / 0               | 0 / 0               | 1 / 1                   | 1 / 1                   | 1 / 1               | 1 / 1               | 1 / 1                       | 1 / 1                            |
| **After 0.159**         | 0 / 0               | 0 / 0               | 0 / 0                   | 0 / 0                   | 0 / 0               | 0 / 0               | 0 / 0                       | 0 / 0                            |

### Explanation:
- **0.05 to 0.09 (Perfect Performance)**: All algorithms perform perfectly with **MRR = 1** and **MAP = 1**.
- **0.09 to 0.159 (ACO & PSO performance degradation)**:
   - **ACO and PSO**: Performance drops to **MRR = 0** and **MAP = 0** after the threshold of 0.09.
   - **Other algorithms (Firefly, Genetic, BAT, ABC, Cuckoo Search, Flower Pollination)**: Continue to have **MRR = 1** and **MAP = 1**.
- **After 0.159 (All algorithms at 0)**: All algorithms see their **MRR = 0** and **MAP = 0** after the relevance threshold exceeds 0.159.

### Key Insights:
- **Threshold Impact**: The relevance threshold impacts the performance of all algorithms, with some (ACO and PSO) showing a drop earlier than others.
- **ACO & PSO**: These two algorithms experience performance degradation at a lower threshold (0.09), while others maintain perfect scores until the threshold exceeds 0.159.
- **Consistent Performance**: Firefly, Genetic, BAT, ABC, Cuckoo Search, and Flower Pollination algorithms maintain a consistent performance across varying relevance thresholds, with perfect precision and recall until the threshold exceeds 0.159.
---
# Legal Document Retrieval using Firefly Algorithm

This project implements a legal document retrieval system using the Firefly Algorithm for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **Firefly Algorithm**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Firefly Algorithm

Initialize fireflies and their light intensities:
```python
num_fireflies = 5
num_iterations = 20
top_k = 3

initial_candidates = np.argsort(similarity_scores)[-top_k:]
fireflies = [list(initial_candidates) for _ in range(num_fireflies)]
light_intensities = [sum(similarity_scores[firefly]) for firefly in fireflies]
```

Update firefly positions based on brighter fireflies:
```python
def update_position(firefly, brighter_firefly):
    new_firefly = list(brighter_firefly)
    while len(new_firefly) < len(firefly):
        candidate = np.argmax(similarity_scores)
        if candidate not in new_firefly:
            new_firefly.append(candidate)
    return new_firefly
```

Optimize firefly positions over iterations:
```python
for iteration in range(num_iterations):
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if light_intensities[j] > light_intensities[i]:
                distance = 1 - np.mean(distance_matrix[fireflies[i], fireflies[j]])
                if distance > 0:
                    fireflies[i] = update_position(fireflies[i], fireflies[j])
                    new_score = sum(similarity_scores[fireflies[i]])
                    if new_score > light_intensities[i]:
                        light_intensities[i] = new_score
```

Output the best firefly per iteration and the final top documents:
```python
best_index = np.argmax(light_intensities)
print(f"Iteration {iteration + 1}: Best Score: {light_intensities[best_index]} | Best Documents: {fireflies[best_index]}")

best_firefly_index = np.argmax(light_intensities)
best_documents = fireflies[best_firefly_index]
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```

---

# Legal Document Retrieval using PSO

This project implements a legal document retrieval system using Particle Swarm Optimization (PSO) for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **PSO**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Particle Swarm Optimization (PSO)

Initialize particles and their velocities:
```python
num_particles = 5
num_iterations = 20
top_k = 3

particles = [np.random.choice(len(similarity_scores), top_k, replace=False) for _ in range(num_particles)]
velocities = [np.zeros(top_k) for _ in range(num_particles)]
personal_best_positions = particles.copy()
personal_best_scores = [sum(similarity_scores[particle]) for particle in particles]
global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
global_best_score = max(personal_best_scores)
```

Update particle positions and velocities:
```python
for iteration in range(num_iterations):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        velocities[i] = 0.5 * velocities[i] + 0.5 * r1 * (personal_best_positions[i] - particles[i]) + 0.5 * r2 * (global_best_position - particles[i])
        particles[i] = np.clip(particles[i] + velocities[i], 0, len(similarity_scores) - 1).astype(int)
        score = sum(similarity_scores[particles[i]])
        if score > personal_best_scores[i]:
            personal_best_positions[i] = particles[i]
            personal_best_scores[i] = score
            if score > global_best_score:
                global_best_position = particles[i]
                global_best_score = score
```

Output the best particle per iteration and the final top documents:
```python
print(f"Iteration {iteration + 1}: Best Score: {global_best_score} | Best Documents: {global_best_position}")

best_documents = global_best_position
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```



---

# Legal Document Retrieval using ACO

This project implements a legal document retrieval system using Ant Colony Optimization (ACO) for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **ACO**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Ant Colony Optimization (ACO)

Initialize ants and pheromone trails:
```python
num_ants = 5
num_iterations = 20
top_k = 3
pheromone_matrix = np.ones((len(similarity_scores), len(similarity_scores)))

def select_next_document(pheromone, similarity, visited):
    probabilities = pheromone * similarity
    probabilities[visited] = 0
    return np.argmax(probabilities)
```

Update pheromone trails based on ant paths:
```python
for iteration in range(num_iterations):
    ant_paths = []
    for _ in range(num_ants):
        path = []
        visited = set()
        for _ in range(top_k):
            next_doc = select_next_document(pheromone_matrix, similarity_scores, visited)
            path.append(next_doc)
            visited.add(next_doc)
        ant_paths.append(path)
        score = sum(similarity_scores[path])
        for doc in path:
            pheromone_matrix[doc] += score

    pheromone_matrix *= 0.9  # Evaporation
```

Output the best ant path per iteration and the final top documents:
```python
best_path = max(ant_paths, key=lambda path: sum(similarity_scores[path]))
print(f"Iteration {iteration + 1}: Best Score: {sum(similarity_scores[best_path])} | Best Documents: {best_path}")

best_documents = best_path
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```

---
# Legal Document Retrieval using Genetic Algorithm

This project implements a legal document retrieval system using the Genetic Algorithm for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **Genetic Algorithm**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Genetic Algorithm

Initialize population and fitness function:
```python
num_individuals = 10
num_generations = 20
top_k = 3

population = [np.random.choice(len(similarity_scores), top_k, replace=False) for _ in range(num_individuals)]

def fitness(individual):
    return sum(similarity_scores[individual])
```

Selection, crossover, and mutation functions:
```python
def selection(population):
    sorted_population = sorted(population, key=fitness, reverse=True)
    return sorted_population[:num_individuals // 2]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, top_k - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(individual):
    mutation_point = np.random.randint(0, top_k)
    individual[mutation_point] = np.random.randint(0, len(similarity_scores))
    return individual
```

Optimize population over generations:
```python
for generation in range(num_generations):
    selected_population = selection(population)
    next_generation = []

    while len(next_generation) < num_individuals:
        parent1, parent2 = np.random.choice(selected_population, 2, replace=False)
        child1, child2 = crossover(parent1, parent2)
        next_generation.append(mutation(child1))
        next_generation.append(mutation(child2))

    population = next_generation

best_individual = max(population, key=fitness)
print(f"Best Score: {fitness(best_individual)} | Best Documents: {best_individual}")

best_documents = best_individual
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```

---
# Legal Document Retrieval using Bat Algorithm

This project implements a legal document retrieval system using the Bat Algorithm for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **Bat Algorithm**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Bat Algorithm

Initialize bats and their velocities:
```python
num_bats = 5
num_iterations = 20
top_k = 3

bats = [np.random.choice(len(similarity_scores), top_k, replace=False) for _ in range(num_bats)]
velocities = [np.zeros(top_k) for _ in range(num_bats)]
frequencies = np.random.rand(num_bats)
loudness = np.ones(num_bats)
pulse_rate = np.ones(num_bats)
best_bat = bats[np.argmax([sum(similarity_scores[bat]) for bat in bats])]
```

Update bat positions and velocities:
```python
for iteration in range(num_iterations):
    for i in range(num_bats):
        frequencies[i] = np.random.rand()
        velocities[i] += (bats[i] - best_bat) * frequencies[i]
        new_bat = np.clip(bats[i] + velocities[i], 0, len(similarity_scores) - 1).astype(int)
        
        if np.random.rand() > pulse_rate[i]:
            new_bat = np.clip(best_bat + 0.001 * np.random.randn(top_k), 0, len(similarity_scores) - 1).astype(int)
        
        if np.random.rand() < loudness[i] and sum(similarity_scores[new_bat]) > sum(similarity_scores[bats[i]]):
            bats[i] = new_bat
            loudness[i] *= 0.9
            pulse_rate[i] *= 0.9
        
        if sum(similarity_scores[bats[i]]) > sum(similarity_scores[best_bat]):
            best_bat = bats[i]
```

Output the best bat per iteration and the final top documents:
```python
print(f"Iteration {iteration + 1}: Best Score: {sum(similarity_scores[best_bat])} | Best Documents: {best_bat}")

best_documents = best_bat
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```
---
# Legal Document Retrieval using Artificial Bee Colony (ABC) Algorithm

This project implements a legal document retrieval system using the Artificial Bee Colony (ABC) Algorithm for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **ABC Algorithm**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Artificial Bee Colony (ABC) Algorithm

Initialize bees and their food sources:
```python
num_bees = 5
num_iterations = 20
top_k = 3

food_sources = [np.random.choice(len(similarity_scores), top_k, replace=False) for _ in range(num_bees)]
fitness_scores = [sum(similarity_scores[food_source]) for food_source in food_sources]
```

Employed bee phase:
```python
for iteration in range(num_iterations):
    for i in range(num_bees):
        new_food_source = food_sources[i].copy()
        k = np.random.randint(0, top_k)
        j = np.random.randint(0, len(similarity_scores))
        new_food_source[k] = j
        new_fitness = sum(similarity_scores[new_food_source])
        if new_fitness > fitness_scores[i]:
            food_sources[i] = new_food_source
            fitness_scores[i] = new_fitness
```

Onlooker bee phase:
```python
probabilities = fitness_scores / np.sum(fitness_scores)
for i in range(num_bees):
    if np.random.rand() < probabilities[i]:
        new_food_source = food_sources[i].copy()
        k = np.random.randint(0, top_k)
        j = np.random.randint(0, len(similarity_scores))
        new_food_source[k] = j
        new_fitness = sum(similarity_scores[new_food_source])
        if new_fitness > fitness_scores[i]:
            food_sources[i] = new_food_source
            fitness_scores[i] = new_fitness
```

Scout bee phase:
```python
for i in range(num_bees):
    if np.random.rand() < 0.1:  # Scout condition
        food_sources[i] = np.random.choice(len(similarity_scores), top_k, replace=False)
        fitness_scores[i] = sum(similarity_scores[food_sources[i]])
```

Output the best food source per iteration and the final top documents:
```python
best_index = np.argmax(fitness_scores)
print(f"Iteration {iteration + 1}: Best Score: {fitness_scores[best_index]} | Best Documents: {food_sources[best_index]}")

best_documents = food_sources[best_index]
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```

---

# Legal Document Retrieval using Flower Pollination Algorithm

This project implements a legal document retrieval system using the Flower Pollination Algorithm for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **Flower Pollination Algorithm**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_flowerpollination.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Flower Pollination Algorithm

Initialize flowers and their positions:
```python
num_flowers = 5
num_iterations = 20
top_k = 3

flowers = [np.random.choice(len(similarity_scores), top_k, replace=False) for _ in range(num_flowers)]
best_flower = flowers[np.argmax([sum(similarity_scores[flower]) for flower in flowers])]
```

Update flower positions based on global and local pollination:
```python
for iteration in range(num_iterations):
    for i in range(num_flowers):
        if np.random.rand() < 0.8:  # Global pollination
            L = np.random.randn(top_k)
            new_flower = np.clip(flowers[i] + L * (best_flower - flowers[i]), 0, len(similarity_scores) - 1).astype(int)
        else:  # Local pollination
            epsilon = np.random.rand()
            j, k = np.random.choice(num_flowers, 2, replace=False)
            new_flower = np.clip(flowers[i] + epsilon * (flowers[j] - flowers[k]), 0, len(similarity_scores) - 1).astype(int)

        if sum(similarity_scores[new_flower]) > sum(similarity_scores[flowers[i]]):
            flowers[i] = new_flower

        if sum(similarity_scores[flowers[i]]) > sum(similarity_scores[best_flower]):
            best_flower = flowers[i]
```

Output the best flower per iteration and the final top documents:
```python
print(f"Iteration {iteration + 1}: Best Score: {sum(similarity_scores[best_flower])} | Best Documents: {best_flower}")

best_documents = best_flower
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```


---

# Legal Document Retrieval using Cuckoo Search Algorithm

This project implements a legal document retrieval system using the Cuckoo Search Algorithm for optimization. The system ranks legal documents based on their relevance to a given query.

## Features

- **TF-IDF Vectorization**: Converts documents and queries into TF-IDF vectors.
- **Cosine Similarity**: Measures the similarity between documents and the query.
- **Cuckoo Search Algorithm**: Optimizes the retrieval of the most relevant documents.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevGO123/legal-reference-analysis.git
   cd legal-reference-analysis
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument.ipynb` to execute the retrieval process.

## Code Overview

### TF-IDF Vectorization

The documents and query are converted into TF-IDF vectors:
```python
document_tfidf_matrix = combined_tfidf_matrix[:-1]
query_tfidf_matrix = combined_tfidf_matrix[-1]
```

### Cosine Similarity

Calculate similarity scores between documents and the query:
```python
similarity_scores = cosine_similarity(document_tfidf_matrix, query_tfidf_matrix).flatten()
```

### Cuckoo Search Algorithm

Initialize nests and their positions:
```python
num_nests = 5
num_iterations = 20
top_k = 3

nests = [np.random.choice(len(similarity_scores), top_k, replace=False) for _ in range(num_nests)]
best_nest = nests[np.argmax([sum(similarity_scores[nest]) for nest in nests])]
```

Update nest positions based on Levy flights and randomization:
```python
for iteration in range(num_iterations):
    for i in range(num_nests):
        step_size = np.random.randn(top_k)
        new_nest = np.clip(nests[i] + step_size * (best_nest - nests[i]), 0, len(similarity_scores) - 1).astype(int)

        if sum(similarity_scores[new_nest]) > sum(similarity_scores[nests[i]]):
            nests[i] = new_nest

        if sum(similarity_scores[nests[i]]) > sum(similarity_scores[best_nest]):
            best_nest = nests[i]

    # Randomize some nests
    for i in range(num_nests):
        if np.random.rand() < 0.25:
            nests[i] = np.random.choice(len(similarity_scores), top_k, replace=False)
```

Output the best nest per iteration and the final top documents:
```python
print(f"Iteration {iteration + 1}: Best Score: {sum(similarity_scores[best_nest])} | Best Documents: {best_nest}")

best_documents = best_nest
print(f"\nTop {top_k} most relevant legal documents for your issue:")
for idx in best_documents:
    print(f"Document {idx}: {legal_documents[idx]}")
```
