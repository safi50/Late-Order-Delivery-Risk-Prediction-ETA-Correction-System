# Late Delivery Risk Prediction & ETA Correction System

A machine learning system for predicting late delivery risk and applying intelligent ETA corrections to improve customer experience on delivery platforms.

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Results](#results)
7. [License](#license)

## Overview

This project addresses late delivery challenges in food delivery services using machine learning. The system consists of two components:

1. **Binary Classification Model**: Predicts orders at high risk of late delivery
2. **Venue-Specific ETA Correction**: Applies intelligent time buffers to high-risk orders

### Key Features

- Binary classification with 78.3% precision at identifying late deliveries
- Venue-specific correction buffers computed from training data
- Risk-based correction system balancing precision and recall
- Comprehensive geospatial analysis using H3 hexagonal indexing
- Evaluation on real-world delivery dataset (Spring 2022, Helsinki)

## Problem Statement

**Goal**: Improve customer experience through more accurate delivery time estimates.

**Challenge**: Optimistic ETA estimates lead to customer disappointment when deliveries are late.

**Solution**: A machine learning model that:
- Identifies orders likely to be delivered late
- Applies venue-specific time buffers to correct ETAs
- Maintains precision above 50% to avoid over-correction

**Impact**:
- Reduced customer complaints
- Improved trust and retention
- Better operational resource allocation
- Competitive advantage through accurate ETAs

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── orders_spring_2022.csv          # Original dataset
│   ├── cleaned_full.csv                # Cleaned dataset with features
│   ├── train.csv                       # Training split (66%)
│   ├── val.csv                         # Validation split (14%)
│   └── test.csv                        # Test split (20%)
│
├── notebooks/
│   ├── data_cleaning_pipeline.ipynb    # Data cleaning and feature engineering
│   ├── exploratory_data_analysis.ipynb # EDA and insights
│   └── model_training.ipynb            # Model training and evaluation
│
└── outputs/
    ├── images/                         # Visualizations
    │   ├── cleaning/
    │   ├── eda/
    │   └── models/
    └── models/                         # Model artifacts and metrics
        ├── model_comparison.csv
        ├── feature_importance.csv
        ├── threshold_comparison.csv
        ├── venue_buffers.json
        ├── model_config.json
        └── test_set_metrics.json
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, LightGBM, XGBoost
- **Visualization**: matplotlib, seaborn
- **Geospatial**: h3, geopandas, shapely
Usage

Execute the notebooks in the following order to reproduce the analysis:

### 1. Data Cleaning & Feature Engineering
```bash
jupyter notebook notebooks/data_cleaning_pipeline.ipynb
```
- Loads raw data from `data/orders_spring_2022.csv`
- Handles missing values, outliers, and duplicates
- Engineers geospatial features (H3 distance, grid distance)
- Creates time-based features (hour, day of week, rush hour)
- Computes venue friction scores
- Outputs: `cleaned_full.csv`, `train.csv`, `val.csv`, `test.csv`

### 2. Exploratory Data Analysis
```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```
- Analyzes delivery time distributions
- Examines ETA calibration
- Investigates temporal, weather, and supply effects
- Performs geospatial analysis with H3 indexing
- Outputs: Visualizations in `outputs/images/eda/`

### 3. Model Training & Evaluation
```bash
jupyter notebook notebooks/model_training.ipynb
```
- Trains multiple classification models
- Computes venue-specific ETA correction buffers
- Evaluates performance on validation and test sets
- Performs threshold optimization
- Outputs: Models and metrics in `outputs/models/`
---

## Results Summary

### Model Performance

**Best Mod

### Model Performance

Best model: LightGBM Classifier with optimized threshold

| Metric | Value |
|--------|-------|
| Precision | 78.3% |
| Recall | 41.7% |
| F1 Score | 54.4% |
| ROC AUC | 0.847 |

### Top Features by Importance

1. `h3_distance_km` - Geographic distance between venue and customer
2. `venue_friction_score` - Historical venue performance metric
3. `distance_x_weather` - Interaction between distance and precipitation
4. `courier_supply_index` - Fleet availability
5. `precipitation` - Weather conditions

### ETA Correction System

- Venue-specific buffers ranging from 2-15 minutes
- Applied when model predicts late risk above threshold
- Computed from training data only (no data leakage)
- Validated on held-out test set

### Business Impact

- High precision (78.3%): Corrections are justified in most cases
- Catches 41.7% of late deliveries proactively
- Reduces unexpected delays without over-correction
- Venue-aware approach accounts for location-specific challenges

## License

This project is licensed under the MIT License - see the LICENSE file for details.