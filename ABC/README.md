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
   cd ABC
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_ABC.ipynb` to execute the retrieval process.

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
