import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Time Series to Image CNN", layout="wide")

st.title("Financial Time Series Classification via Gramian Angular Fields")
st.write(
    "Minor Project: Converting 1D non-stationary stock series into 2D spatial "
    "representations (GASF) for trend classification using a 2D Convolutional Neural Network."
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Select Asset", ["NVDA", "AAPL", "MSFT", "GOOGL"], index=0)
window_size = st.sidebar.slider("Lookback Window (Days)", min_value=15, max_value=40, value=20)
barrier_pct = st.sidebar.slider("Target Horizon Threshold (%)", min_value=1.0, max_value=3.5, value=2.0) / 100.0
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=35, value=20)
batch_size = 32

# ---------------------------------------------------------
# 1. Mathematical Transformation: Gramian Angular Field
# ---------------------------------------------------------
def create_gasf(series_window):
    """
    Transforms a 1D window of normalized prices into a 2D Gramian Angular Summation Field.
    Formula: GASF_(i,j) = cos(phi_i + phi_j) = x_i * x_j - sqrt(1 - x_i^2) * sqrt(1 - x_j^2)
    """
    min_val = np.min(series_window)
    max_val = np.max(series_window)
    if max_val - min_val == 0:
        scaled = np.zeros_like(series_window)
    else:
        scaled = 2 * (series_window - min_val) / (max_val - min_val) - 1
    
    scaled = np.clip(scaled, -1.0, 1.0)
    
    cos_phi = scaled
    sin_phi = np.sqrt(np.maximum(0.0, 1.0 - scaled ** 2))
    
    gasf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
    return gasf

# ---------------------------------------------------------
# 2. Data Fetching & Triple-Barrier Labeling
# ---------------------------------------------------------
def load_and_prepare_data(symbol, win_size, threshold):
    df = yf.download(symbol, period="3y", interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df['Close'].iloc[:, 0].dropna()
    elif 'Close' in df.columns:
        close_series = df['Close'].dropna()
    else:
        close_series = df.iloc[:, 0].dropna()
        
    prices = close_series.values
    dates = close_series.index
    
    horizon = 5
    images = []
    labels = []
    valid_dates = []

    for i in range(win_size, len(prices) - horizon):
        window = prices[i - win_size : i]
        future = prices[i : i + horizon]
        
        pct = (future - prices[i - 1]) / prices[i - 1]
        hit_up = np.where(pct >= threshold)[0]
        hit_down = np.where(pct <= -threshold)[0]
        
        first_up = hit_up[0] if len(hit_up) > 0 else 999
        first_down = hit_down[0] if len(hit_down) > 0 else 999
        
        if first_up < first_down:
            label = 2  # Long breakout
        elif first_down < first_up:
            label = 0  # Stop loss trigger
        else:
            label = 1  # Consolidation
            
        gasf_matrix = create_gasf(window)
        images.append(gasf_matrix)
        labels.append(label)
        valid_dates.append(dates[i])

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    images = np.expand_dims(images, axis=1)  # Shape: (N, 1, H, W)
    
    return close_series, images, labels, valid_dates

# ---------------------------------------------------------
# 3. 2D Convolutional Neural Network
# ---------------------------------------------------------
class TimeSeriesCNN(nn.Module):
    def __init__(self, input_dim):
        super(TimeSeriesCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# ---------------------------------------------------------
# Execution Section
# ---------------------------------------------------------
if st.button("Run Pipeline & Train Model"):
    with st.spinner("Downloading price series and generating Gramian Angular Fields..."):
        close_series, images, labels, valid_dates = load_and_prepare_data(ticker, window_size, barrier_pct)

    st.success(f"Generated {images.shape[0]} samples. Image shape: {images.shape[2]}x{images.shape[3]} pixels.")

    st.subheader("1. Time Series to 2D Image Transformation")
    col1, col2 = st.columns(2)
    
    sample_idx = -10
    sample_window = close_series.loc[:valid_dates[sample_idx]].iloc[-window_size:]
    sample_img = images[sample_idx, 0]

    with col1:
        fig_ts, ax_ts = plt.subplots(figsize=(5, 3))
        ax_ts.plot(sample_window.values, marker='o', markersize=3, color='#1f77b4')
        ax_ts.set_title(f"1D Price Sequence ({window_size} Days)")
        ax_ts.set_xlabel("Time Step (t)")
        ax_ts.set_ylabel("Price")
        ax_ts.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_ts)

    with col2:
        fig_img, ax_img = plt.subplots(figsize=(5, 3))
        cax = ax_img.imshow(sample_img, cmap='Spectral_r', origin='lower')
        fig_img.colorbar(cax, ax=ax_img)
        ax_img.set_title("Encoded 2D Gramian Angular Field (GASF)")
        ax_img.set_xlabel("t_i")
        ax_img.set_ylabel("t_j")
        st.pyplot(fig_img)

    st.subheader("2. Model Training & Evaluation")
    split = int(len(labels) * 0.8)
    
    X_train = torch.tensor(images[:split])
    y_train = torch.tensor(labels[:split])
    X_test = torch.tensor(images[split:])
    y_test = torch.tensor(labels[split:])
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = TimeSeriesCNN(input_dim=window_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    loss_history = []
    with st.spinner("Training 2D CNN across epochs..."):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)
            
            epoch_loss = running_loss / len(train_dataset)
            loss_history.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        preds = torch.argmax(test_outputs, dim=1).numpy()
        y_test_np = y_test.numpy()
        test_acc = (preds == y_test_np).mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Out-of-Sample Accuracy", f"{test_acc * 100:.2f}%")
    m2.metric("Dataset Split", f"{split} Train / {len(y_test)} Test")
    m3.metric("Baseline Random Guess", "33.33%")

    col3, col4 = st.columns(2)
    with col3:
        fig_loss, ax_loss = plt.subplots(figsize=(5, 3.5))
        ax_loss.plot(range(1, epochs + 1), loss_history, marker='.', color='#ff7f0e')
        ax_loss.set_title("Training Loss Convergence")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Cross Entropy Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_loss)

    with col4:
        st.write("**Confusion Matrix**")
        class_names = ["Stop-Loss (-)", "Consolidation (0)", "Breakout (+)"]
        cm = confusion_matrix(y_test_np, preds)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
        cax_cm = ax_cm.imshow(cm, cmap='Blues')
        fig_cm.colorbar(cax_cm, ax=ax_cm)
        ax_cm.set_xticks(range(3))
        ax_cm.set_yticks(range(3))
        ax_cm.set_xticklabels(class_names, rotation=20)
        ax_cm.set_yticklabels(class_names)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        
        for r in range(3):
            for c in range(3):
                ax_cm.text(c, r, str(cm[r, c]), ha="center", va="center", color="black" if cm[r, c] < cm.max()/2 else "white")
        st.pyplot(fig_cm)

    st.subheader("3. Classification Performance Summary")
    rep = classification_report(y_test_np, preds, target_names=class_names, output_dict=True, zero_division=0)
    st.table(pd.DataFrame(rep).transpose().style.format(precision=3))