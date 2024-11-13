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
   cd CuckooSearch
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_CSA.ipynb` to execute the retrieval process.

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
