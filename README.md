# SASNet: LLM Embeddings and Siamese Transformer Network for Recommendation Systems

## Overview

SASNet is a novel approach to recommendation systems that leverages Large Language Models (LLMs) and a Siamese Transformer-based architecture. This project aims to address common challenges in recommendation systems, including overspecialization, the cold-start problem, and the need for in-depth item knowledge.

## Table of Contents

## Project Structure

The project is organized into the following main directories:

- `data/`: Contains the dataset and embeddings
- `src/`: Source code for the models and data processing
- `train/`: Scripts for training the models
- `results/`: Stores evaluation results, plots, and graphs
- `inference/`: Scripts for running inference on trained models

## Key Features

- Utilizes quantized LLM (Phi 3 3.8B-Q4-GPTQ) for rich embeddings
- Custom Siamese AttentionSetNet (SASNet) architecture
- Dynamic embedding generation for users and activities

## Dataset and Embeddings

The embeddings dataset can be downloaded from: [PLACEHOLDER_LINK]

The dataset is in parquet format and contains pre-computed embeddings for the Yelp reviews.

## Models

1. Hulk3: SVM classifier using embeddings from Phi-3
2. huLLK: SVM classifier using embeddings from sentence-transformer miniLM
3. SiameseAttentionSetNet: Custom transformer architecture using embeddings from Phi-3
4. SiameseAttentionSetNet + miniLM: Custom transformer architecture using embeddings from sentence-transformer miniLM (planned)

## Installation

```console
pip install sasnet
```

## Usage

### Training

To train the models, use the Jupyter notebooks in the `train/` directory. You can modify the following parameters at the start of the notebook:

```python
SAMPLE_SIZE = 5  # Recommended range: 3-10
BATCH_SIZE = 512  # Recommended range: 128-1024
LLM_dim = 3072  # Fixed for Phi-3
hidden_dim = 256  # Recommended range: 128-512
num_heads = 4  # Recommended range: 2-8
ffn_dim = 1024  # Recommended range: 512-2048
dropout_rate = 0.2  # Recommended range: 0.1-0.5
n_classes = 3  # Fixed for this classification task

# Training parameters
EPOCHS = 3000  # Recommended range: 1000-5000
PATIENCE = 150  # Recommended range: 50-200
```

### Evaluation

We use custom metrics to evaluate both the embeddings and the network performance:

#### Embedding Evaluation

We assess the quality of embeddings using two key metrics:

1. Sentiment Analysis: Measures cosine similarity between "bad" and "excellent" embeddings.
2. Category Understanding: Compares embeddings of different business categories.

The overall embedding score is calculated as:

```
s = (-0.25 * NMSE_c) + (0.25 * NPR_c) - (0.5 * NAE_s)
```

Where:
- NMSE_c: Normalized Mean Squared Error for category understanding
- NPR_c: Normalized Pearson's Correlation for category understanding
- NAE_s: Normalized Absolute Error for sentiment analysis

#### Network Evaluation

The network is evaluated using standard classification metrics:
- Accuracy
- F1-score

## Results

- LLM embeddings captured greater semantic similarity than state-of-the-art sentence embedders.
- SVM with LLM embeddings outperformed the TF-IDF+SVM baseline by 12.25%.
- SASNet performed 5% worse than the TF-IDF+SVM baseline on the current dataset.

## Visualization

Training progress and network architecture visualizations can be found in the `results/plots/` directory. These are generated during the training process in the SiameseAttentionSetNet notebook.

## Contributing

We welcome contributions to the SASNet project. If you'd like to contribute, please:

1. Fork the repository
2. Create a new branch for your feature
3. Implement your changes
4. Submit a pull request

For major changes or new features, please open an issue first to discuss the proposed changes.

## License

`sasnet` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Contact

[Add contact information for the project maintainers]

## Acknowledgements

We would like to thank the Yelp Open Dataset for providing the data used in this research.

## Citation

If you use this work in your research, please cite:

```
Amer, A., Çabuk, B., Chinello, F., & Rotov, D. (2024). SASNet: Dynamic LLM Embeddings and Combinatorial Training for Recommendation Systems.
```
