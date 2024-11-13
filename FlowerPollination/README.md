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
   cd FlowerPollination
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_FPA.ipynb` to execute the retrieval process.

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