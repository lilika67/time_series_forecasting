# Beijing Air Quality Forecasting

This project implements deep learning models to forecast PM2.5 air pollution levels in Beijing using historical weather and air quality data.

## Project Overview

Air pollution is a significant environmental and health concern in major urban centers like Beijing. This project uses historical meteorological and air quality data to build predictive models for PM2.5 pollution levels. By leveraging recurrent neural networks and convolutional neural networks, we create forecasting models that can provide accurate predictions to help inform public health decisions.

## Dataset

The dataset used in this project contains hourly measurements of various meteorological features and air pollutants for Beijing from 2013-2017, including:

- **DEWP**: Dew Point
- **TEMP**: Temperature
- **PRES**: Pressure
- **Iws**: Cumulated wind speed
- **Is**: Cumulated hours of snow
- **Ir**: Cumulated hours of rain
- **cbwd**: Combined wind direction
- **PM2.5**: PM2.5 concentration (target variable)

The data is split into training (2013-2017) and test sets (2017-2018) for model evaluation.

## Model Architecture

We implemented and compared several deep learning architectures:

1. **Baseline LSTM Model**: Two-layer LSTM network with dropout for regularization
2. **CNN-LSTM Hybrid Model**: Convolutional layers followed by LSTM layers to capture both local and temporal patterns
3. **Deeper LSTM Model**: Three-layer LSTM with higher capacity for learning complex temporal dependencies

Each model uses a sequence of 24 hours (lookback window) to predict the next hour's PM2.5 value.

## Requirements

To run this project, you need the following dependencies:

```
tensorflow>=2.0.0
pandas
numpy
seaborn
matplotlib
scikit-learn
```

## Instructions for Reproducing Results

Follow these steps to reproduce our results:

1. **Data Preparation**:
   - Download the dataset from Kaggle(https://www.kaggle.com/competitions/assignment-1-time-series-forecasting-may-2025/data) or use the provided files (`train.csv`, `test.csv`)
   - Place the files in a directory accessible to your notebook

2. **Setup Environment**:
   ```
   pip install tensorflow pandas numpy seaborn matplotlib scikit-learn
   ```

3. **Data Preprocessing**:
   - The notebook includes preprocessing steps for handling missing values, feature scaling, and sequence creation
   - Ensure you run all the preprocessing cells before model training

4. **Model Training**:
   - Run the model training cells for each architecture (Baseline, CNN-LSTM, Deeper LSTM)
   - Model parameters can be adjusted in the respective functions

5. **Generating Predictions**:
   - After training, models will automatically generate prediction files
   - Prediction files follow Kaggle submission format

6. **Evaluation**:
   - Model performance is evaluated using RMSE on the validation set
   - Training history plots show loss curves for training and validation

## Implementation Details

### Data Preprocessing

- Missing values in PM2.5 are imputed with mean values
- Temporal features (hour, day, month, etc.) are extracted from the datetime index
- Cyclical encoding of time features using sine and cosine transformations
- Min-max scaling is applied to numerical features
- Lag features are created to capture temporal dependencies

### Feature Engineering

The following features are engineered to improve model performance:

- **Lag Features**: Previous 1-hour and 2-hour PM2.5 values
- **Rolling Statistics**: 3-hour rolling mean of PM2.5
- **Cyclical Time Features**: Hour of day and day of week encoded using sine and cosine
- **Wind Direction**: One-hot encoded wind direction categories

### Sequence Creation

Time series data is transformed into sequences for model input:

- Each input sequence consists of 24 time steps (hours)
- Each time step contains all features (weather variables and engineered features)
- Target is the PM2.5 value at the next time step after the sequence

### Model Training

All models are trained with:

- Adam optimizer with learning rate of 0.001
- Mean Squared Error loss function
- Early stopping with patience to prevent overfitting
- 20% validation split for monitoring performance


## Future Improvements

-Feature engineering to capture seasonal patterns
-Experiment with bidirectional LSTM layers
-Implement attention mechanisms for better temporal focus


## References

- [Kaggle  Air-Quality Competition]([https://www.kaggle.com/competitions/](https://www.kaggle.com/competitions/assignment-1-time-series-forecasting-may-2025/data))
