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
   cd Firefly
   ```

2. Install the required packages:
   ```bash
   pip install numpy scikit-learn
   ```

## Usage

1. Prepare your legal documents and query.
2. Run the Jupyter notebook `legaldocument_firefly.ipynb` to execute the retrieval process.

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
