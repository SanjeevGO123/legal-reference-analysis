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
   cd ACO
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_ACO.ipynb` to execute the retrieval process.

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