\# Financial Time Series Classification Using Gramian Angular Fields and 2D CNNs



A machine learning project exploring the conversion of 1D financial time series into 2D spatial image matrices using Gramian Angular Summation Fields (GASF), classified via a Convolutional Neural Network.



\## Problem Statement

Standard machine learning approaches to stock forecasting rely on 1D recurrent models (LSTMs, GRUs) minimizing Mean Squared Error on closing prices. In practice, this often creates a model that simply copies the previous day's price with a 1-day lag. 



This project explores a computer vision alternative:

1\. Encode continuous price windows into 2D polar matrices using trigonometric transformations.

2\. Allow a 2D CNN to learn visual geometric textures and trend consolidation patterns.

3\. Classify forward movements using the Triple-Barrier Method (+2% take-profit vs -2% stop-loss vs flat consolidation over a 5-day horizon).



\## Mathematical Formulation

For a time series $X = \\{x\_1, x\_2, \\dots, x\_n\\}$, the values are rescaled to $\[-1, 1]$:

$$\\tilde{x}\_i = \\frac{(x\_i - \\max(X)) + (x\_i - \\min(X))}{\\max(X) - \\min(X)}$$



Values are mapped to polar coordinates by encoding the value as the cosine of the angle:

$$\\phi = \\arccos(\\tilde{x}\_i), \\quad r = \\frac{t\_i}{N}$$



The Gramian Angular Summation Field (GASF) is computed by:

$$\\text{GASF}\_{i,j} = \\cos(\\phi\_i + \\phi\_j) = \\tilde{x}\_i \\tilde{x}\_j - \\sqrt{1 - \\tilde{x}\_i^2}\\sqrt{1 - \\tilde{x}\_j^2}$$



This results in an $N \\times N$ matrix preserving temporal dependencies across all lag combinations.



\## Architecture

\- \*\*Data Ingestion:\*\* Daily OHLCV data fetched via `yfinance`.

\- \*\*Pre-processing:\*\* Rolling window normalization and vectorized GASF matrix creation.

\- \*\*Model:\*\* 2D CNN with 2 Conv2D layers, Batch Normalization, Max Pooling, and a Fully Connected classification head.

\- \*\*Evaluation:\*\* Walk-forward out-of-sample chronological split (80% train / 20% test).



\## Requirements

