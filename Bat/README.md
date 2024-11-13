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
   cd Bat
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_bat.ipynb` to execute the retrieval process.

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