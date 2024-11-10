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