This repository contains my implementation of the KDD 2025 paper *“Generative Next POI Recommendation with Semantic ID.”*

The following components are already available:

### **RQ-VAE Module**

The core implementation of the Residual Vector Quantized Variational Autoencoder (RQ-VAE) is provided.

* 使用 `code/train_rqvae.py` to train a custom codebook.
* After training, use `codebook.py` to generate the mapping from discrete token IDs to semantic IDs.

### **Sample Dataset**

A sample dataset based on the **NYC** check-in data is included for demonstration and evaluation.

### **Data Preprocessing**

The data preprocessing pipeline is available in the following Jupyter notebooks:

* `dataprocess.ipynb`: raw data cleaning and formatting
* `data2json.ipynb`: conversion of processed data into model-ready JSON format

### **Model Fine-tuning**

using QWEN to achieve LLM recommendation





