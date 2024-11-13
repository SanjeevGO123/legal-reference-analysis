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
   cd Genetic
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_GA.ipynb` to execute the retrieval process.

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