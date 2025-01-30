<!-- developer docs -->
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://n1developer-ubt.github.io/data-preprocessing-juml/dev/)

[![codecov](https://codecov.io/gh/n1developer-ubt/data-preprocessing-juml/branch/main/graph/badge.svg)](https://codecov.io/gh/n1developer-ubt/data-preprocessing-juml)

# Julia Data Preprocessing Pipeline
Created as part of the JuML course at TU Berlin - Group F (2024-25). This project is inspired by scikit-learns [dataset transformations](https://scikit-learn.org/stable/data_transforms.html) library and implements some of its functions in Julia. 

Main Contributors:
- Usama Bin Tariq
- Maria Omotayo Mamie Cole
- Alexander Smirnov
- Cedric Braun

## About
This package is meant for building robust data preprocessing pipelines. It streamlines workflows by offering modules for sequential data transformations, feature extraction, preprocessing, and handling missing values, ensuring clean and consistent datasets for machine learning and analysis.

### Pipeline
The `PipelineModule` enables chaining of data transformation and modeling steps into a unified workflow. It ensures consistent and sequential application of preprocessing and estimation, simplifying the process of fitting, transforming, and predicting on datasets. For more details see the [pipeline documentation](docs/pipeline.md).

### Feature Extraction
The `FeatureExtraction` module provides tools for converting raw data into structured formats suitable for analysis. It includes utilities for vectorization, dimensionality reduction, polynomial feature generation, and scaling to extract meaningful representations from raw datasets.

### Preprocessing
The `Preprocessing` module offers a suite of transformations to prepare data for analysis, including scaling, normalization, encoding categorical variables, and binarization. These tools ensure data is clean and standardized, ready for modeling.

### Handling Missing Values
The `MissingValue` module provides strategies for dealing with incomplete datasets. It includes methods for imputation, dropping missing values, and filling missing entries with constants or computed values like mean or median, ensuring robust data pipelines.


## Getting started
-  see [DOCS](https://n1developer-ubt.github.io/data-preprocessing-juml/dev/)
