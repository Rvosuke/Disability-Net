# Disability-Net: A Causality-Based Disability Early Warning Model Using Longitudinal Data 🏥👵👴

![GitHub stars](https://img.shields.io/github/stars/Rvosuke/Disability-Net?style=social)
![GitHub forks](https://img.shields.io/github/forks/Rvosuke/Disability-Net?style=social)
![GitHub issues](https://img.shields.io/github/issues/Rvosuke/Disability-Net)
![GitHub license](https://img.shields.io/github/license/Rvosuke/Disability-Net)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [File Descriptions](#file-descriptions)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Data Availability](#data-availability)
- [Additional Resources](#additional-resources)
- [Publication Status](#publication-status)

## 🔍 Project Overview

This project implements a causality-based disability early warning model using longitudinal data. It combines causal discovery techniques with graph neural networks for classification tasks. The model is designed to identify potential disability risks in elderly populations, utilizing advanced machine learning and causal inference methods.

## 📁 File Descriptions

- `multiple_interpolation.py`: Source code for multiple imputation based on LightGBM, used in the Chinese Elderly Disability Dataset section.
- `NOTEARS.py`: Implementation of causal discovery and causal graph post-processing.
- `Causal-GNN.py`: Core implementation of the Causal Graph Neural Network.
- `feature_selection.py`: Contains implementations of mRMR and CIFE feature selection methods.
- `decision.py`: Code for plotting clinical benefit curves.
- `load_datasets.py`: Script for data preparation and processing.
- `main.py`: Main script to run the classification task.

## 🛠 Dependencies

The project dependencies are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Prepare your dataset or use the default breast cancer dataset from scikit-learn.
2. Run `load_datasets.py` to perform causal discovery and generate necessary CSV files:
   - `expression.csv`: Sample feature matrix
   - `target.csv`: Label vector
   - `adjacency_matrix.csv`: Adjacency matrix
3. Execute `main.py` to start the classification task and obtain results.

You can adjust parameters in `main.py` to fine-tune the algorithm's performance:
- `no_pos`: Controls whether positional encoding is used
- `gcn_base_layers`: Sets the number of base layers in the graph convolution network

## 📊 Data Availability

While the primary data used in this study is not publicly available, all code can be run using public datasets such as the breast cancer dataset from scikit-learn or custom datasets generated using `make_classification`.

## 📚 Additional Resources

For more information on causal discovery techniques, please refer to the CausalNex documentation:
[CausalNex Documentation](https://causalnex.readthedocs.io/en/latest/index.html)

## 📝 Publication Status

The research paper associated with this project is currently under review at npj Digital Medicine (part of the Nature Partner Journals series). Please note that the status of the paper may change, and we will update this section accordingly.

---

📣 We welcome contributions and feedback! If you have any questions or suggestions, please open an issue or submit a pull request.

