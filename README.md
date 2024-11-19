# Algorithm Performance Comparison (MRR and MAP for Varying Relevance Thresholds)

| **Relevance Threshold** | **ACO (MRR & MAP)** | **PSO (MRR & MAP)** | **Firefly (MRR & MAP)** | **Genetic (MRR & MAP)** | **BAT (MRR & MAP)** | **ABC (MRR & MAP)** | **Cuckoo Search (MRR & MAP)** | **Flower Pollination (MRR & MAP)** |
|-------------------------|---------------------|---------------------|-------------------------|-------------------------|---------------------|---------------------|-----------------------------|----------------------------------|
| **0.01 to 0.09**        | 1 / 1               | 1 / 1               | 1 / 1                   | 1 / 1                   | 1 / 1               | 1 / 1               | 1 / 1                       | 1 / 1                            |
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


