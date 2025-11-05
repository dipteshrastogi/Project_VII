#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend so plots always save (works in headless env)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 2025
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# ------------------------------
# LOAD DATASET FROM CSV
# ------------------------------
def load_dataset_from_csv(input_dir='./dataset', prefix='channel_data'):
    """
    Load the dataset from CSV files created by Part 1.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files
    prefix : str
        Prefix of CSV filenames
    
    Returns:
    --------
    X : numpy array
        Input features (n_samples, sequence_length, n_features)
    y : numpy array
        Target outputs (n_samples, output_dim)
    rho_values : numpy array
        Correlation coefficients (n_samples,)
    """
    print("\n" + "="*60)
    print("LOADING DATASET FROM CSV")
    print("="*60)
    
    # Load shape information
    shape_path = os.path.join(input_dir, f'{prefix}_shape_info.csv')
    if not os.path.exists(shape_path):
        raise FileNotFoundError(f"Shape info file not found: {shape_path}")
    
    shape_df = pd.read_csv(shape_path)
    n_samples = int(shape_df['n_samples'].values[0])
    seq_len = int(shape_df['sequence_length'].values[0])
    n_features = int(shape_df['n_features'].values[0])
    output_dim = int(shape_df['output_dim'].values[0])
    
    print(f"Dataset dimensions:")
    print(f"  Samples: {n_samples}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features per timestep: {n_features}")
    print(f"  Output dimension: {output_dim}")
    
    # Load X (input sequences)
    X_path = os.path.join(input_dir, f'{prefix}_X.csv')
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Input file not found: {X_path}")
    
    X_df = pd.read_csv(X_path)
    X_flattened = X_df.values  # (n_samples, seq_len * n_features)
    X = X_flattened.reshape(n_samples, seq_len, n_features)
    print(f"✓ Loaded X from: {X_path}")
    print(f"  Shape: {X.shape}")
    
    # Load y (target outputs)
    y_path = os.path.join(input_dir, f'{prefix}_y.csv')
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Target file not found: {y_path}")
    
    y_df = pd.read_csv(y_path)
    y = y_df.values
    print(f"✓ Loaded y from: {y_path}")
    print(f"  Shape: {y.shape}")
    
    # Load metadata (rho values)
    metadata_path = os.path.join(input_dir, f'{prefix}_metadata.csv')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata_df = pd.read_csv(metadata_path)
    rho_values = metadata_df['rho'].values
    print(f"✓ Loaded metadata from: {metadata_path}")
    
    print("\n" + "="*60)
    print("DATASET LOADED SUCCESSFULLY!")
    print("="*60)
    
    return X, y, rho_values

# ------------------------------
# NEW: Plot Data Physics
# ------------------------------
def plot_data_physics(X, y, rho_values, n_antennas, savedir='.'):
    """
    Plots the underlying data physics by comparing h(t+1) vs h(t).
    This visualizes the Gauss-Markov model: h(t+1) = rho*h(t) + noise
    """
    print("\n" + "="*60)
    print("ANALYZING DATA PHYSICS (h(t+1) vs h(t))")
    print("="*60)

    # Get the last timestep of the input sequence: h(t)
    # X shape is (n_samples, seq_len, n_features)
    # We want the last time step: X[:, -1, :]
    h_t = X[:, -1, :]
    
    # The target y is h(t+1)
    h_tplus1 = y
    
    # We'll plot just Antenna 0 for simplicity
    # h_t features: [real_0...real_N, imag_0...imag_N, rho]
    h_t_real_ant0 = h_t[:, 0]
    # h_tplus1 features: [real_0...real_N, imag_0...imag_N]
    h_tplus1_real_ant0 = h_tplus1[:, 0]

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'h_t_real': h_t_real_ant0,
        'h_tplus1_real': h_tplus1_real_ant0,
        'rho': rho_values
    })

    # Create 4 bins for rho to plot them separately
    bins = [0.6, 0.7, 0.8, 0.9, 0.95]
    bin_labels = [
        'Rho: 0.6 - 0.7',
        'Rho: 0.7 - 0.8',
        'Rho: 0.8 - 0.9',
        'Rho: 0.9 - 0.95'
    ]
    df['rho_bin'] = pd.cut(df['rho'], bins=bins, labels=bin_labels, include_lowest=True)

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Data Physics: h(t+1) vs. h(t) for different Rho\n(Antenna 0, Real Part)',
                 fontsize=16, fontweight='bold')

    # Iterate through each bin and its corresponding subplot axis
    for (bin_name, group), ax in zip(df.groupby('rho_bin', observed=True), axes.flatten()):
        
        # Scatter plot of the data
        ax.scatter(group['h_t_real'], group['h_tplus1_real'], s=5, alpha=0.1)
        
        # Calculate the ideal line
        # We'll use the mean rho in this bin as the slope
        rho_mean = group['rho'].mean()
        
        # Get plot limits to draw the line
        x_lim = ax.get_xlim()
        x_line = np.array(x_lim)
        y_line = rho_mean * x_line
        
        # Plot the ideal line
        ax.plot(x_line, y_line, 'r--', linewidth=2, 
                label=f'Ideal Slope (rho ≈ {rho_mean:.2f})')
        
        ax.set_title(bin_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('h(t) - Real Part', fontsize=11)
        ax.set_ylabel('h(t+1) - Real Part', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()
        ax.set_aspect('equal') # Make axes equal to see slope correctly

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = os.path.join(savedir, 'data_physics_h_vs_h.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved data physics plot to: {p}")
    print("="*60)


# ------------------------------
# STEP 2: Preprocessing
# ------------------------------
def preprocess_data(X, y, rho_values, test_size=0.3, val_size=0.5, random_state=SEED):
    """
    Splits into train/val/test and scales features & targets.
    Returns scaled partitions and fitted scalers.
    """
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)

    # Split rho_values alongside X and y
    X_train, X_temp, y_train, y_temp, rho_train, rho_temp = train_test_split(
        X, y, rho_values, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test, rho_val, rho_test = train_test_split(
        X_temp, y_temp, rho_temp, test_size=val_size, random_state=random_state
    )

    print(f"Data split:")
    print(f"  Training: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

    # flatten time dimension for scaling features (fit scaler on training features)
    n_samples, seq_len, n_features = X_train.shape
    train_features_reshaped = X_train.reshape(-1, n_features)
    val_features_reshaped = X_val.reshape(-1, n_features)
    test_features_reshaped = X_test.reshape(-1, n_features)

    scaler_X = StandardScaler()
    train_scaled = scaler_X.fit_transform(train_features_reshaped).reshape(X_train.shape)
    val_scaled = scaler_X.transform(val_features_reshaped).reshape(X_val.shape)
    test_scaled = scaler_X.transform(test_features_reshaped).reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"\nNormalization complete!")
    print(f"  Mean of training features: {train_scaled.mean():.6f}")
    print(f"  Std of training features: {train_scaled.std():.6f}")

    # Return the split rho values
    return (train_scaled, y_train_scaled, val_scaled, y_val_scaled, test_scaled, y_test_scaled,
            rho_train, rho_val, rho_test, scaler_X, scaler_y)


# ------------------------------
# STEP 3: Model
# ------------------------------
def build_lstm_model(input_shape, output_dim, lstm_units=64, dense_units=32, dropout=0.2):
    """
    Build LSTM-based neural network for channel prediction.
    
    Architecture:
    Input → LSTM(64) → Dense(32) → Dropout → Output
    """
    print("\n" + "="*60)
    print("BUILDING NEURAL NETWORK MODEL")
    print("="*60)
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units, name='lstm'),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(output_dim, activation='linear')
    ], name='Channel_Predictor')
    
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    
    print("\nModel architecture:")
    model.summary()
    
    return model


# ------------------------------
# STEP 4: Train
# ------------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, savedir='.'):
    """
    Train the neural network model.
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(savedir, 'best_model.keras'), monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print(f"\nStarting training...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
    
    print(f"\nTraining complete!")
    print(f"  Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history


# ------------------------------
# STEP 5: Robust evaluation and plotting (all saved to disk)
# ------------------------------
def evaluate_model_and_plot_all(model, X_test, y_test_scaled, rho_test, scaler_y, n_antennas,
                                n_display=5, savedir='.'):
    """
    Evaluate model performance and create all visualizations.
    """
    print("\n" + "="*60)
    print("EVALUATING & PLOTTING")
    print("="*60)

    # predict and inverse transform
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test_scaled)

    # scalar metrics
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    # reconstruct complex arrays
    y_true_c = y_true[:, :n_antennas] + 1j * y_true[:, n_antennas:]
    y_pred_c = y_pred[:, :n_antennas] + 1j * y_pred[:, n_antennas:]

    per_ant_mse = np.mean(np.abs(y_true_c - y_pred_c)**2, axis=0)
    per_ant_power = np.mean(np.abs(y_true_c)**2, axis=0)
    per_ant_nmse = per_ant_mse / (per_ant_power + 1e-12)
    per_ant_nmse_db = 10*np.log10(per_ant_nmse + 1e-12)

    overall_nmse = np.sum(per_ant_mse) / (np.sum(per_ant_power) + 1e-12)
    overall_nmse_db = 10*np.log10(overall_nmse + 1e-12)

    print(f"\nTest Set Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  NMSE: {overall_nmse:.6e} ({overall_nmse_db:.2f} dB)")
    print(f"  Per-antenna NMSE (dB): {np.round(per_ant_nmse_db, 3)}")

    # Calculate per-sample NMSE for rho plotting
    sample_error_power = np.sum(np.abs(y_true_c - y_pred_c)**2, axis=1) # (n_samples,)
    sample_true_power = np.sum(np.abs(y_true_c)**2, axis=1) # (n_samples,)
    sample_nmse = sample_error_power / (sample_true_power + 1e-12)
    sample_nmse_db = 10 * np.log10(sample_nmse + 1e-12)
    
    # Create a DataFrame for easy grouping
    eval_df = pd.DataFrame({
        'rho': rho_test,
        'nmse_db': sample_nmse_db
    })
    
    # Create 10 bins from min to max rho
    n_bins = 10
    bins = np.linspace(eval_df['rho'].min(), eval_df['rho'].max(), n_bins + 1)
    eval_df['rho_bin_labels'] = pd.cut(eval_df['rho'], bins=bins, include_lowest=True)
    
    # Calculate mean NMSE per bin
    binned_nmse = eval_df.groupby('rho_bin_labels')['nmse_db'].mean()
    # Get the center of each bin for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    metrics = {
        'mse': mse, 'rmse': rmse, 'mae': mae,
        'overall_nmse': overall_nmse, 'overall_nmse_db': overall_nmse_db,
        'per_ant_nmse_db': per_ant_nmse_db, 'per_ant_nmse': per_ant_nmse
    }

    # ----- Plot 1: sample real/imag comparison -----
    n_display = min(n_display, y_true.shape[0])
    fig, axs = plt.subplots(n_display, 2, figsize=(14, 3*n_display))
    for i in range(n_display):
        axr = axs[i,0] if n_display>1 else axs[0]
        axi = axs[i,1] if n_display>1 else axs[1]
        axr.plot(y_true[i, :n_antennas], 'o-', label='True (real)', linewidth=2)
        axr.plot(y_pred[i, :n_antennas], 's--', label='Pred (real)', linewidth=2)
        axr.set_title(f"Sample {i+1} - Real Parts", fontweight='bold')
        axr.legend(); axr.grid(alpha=0.3)

        axi.plot(y_true[i, n_antennas:], 'o-', label='True (imag)', linewidth=2)
        axi.plot(y_pred[i, n_antennas:], 's--', label='Pred (imag)', linewidth=2)
        axi.set_title(f"Sample {i+1} - Imag Parts", fontweight='bold')
        axi.legend(); axi.grid(alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(savedir, 'predictions_samples.png')
    fig.savefig(p1, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"\n✓ Saved: {p1}")

    # ----- Plot 2: per-antenna NMSE (dB) -----
    fig = plt.figure(figsize=(8,4))
    ants = np.arange(1, n_antennas+1)
    plt.bar(ants, metrics['per_ant_nmse_db'], color='steelblue', edgecolor='black')
    plt.xlabel('Antenna index', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title('Per-Antenna NMSE (dB)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(savedir, 'per_antenna_nmse.png')
    fig.savefig(p2, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"✓ Saved: {p2}")

    # ----- Plot 3: error histogram & true vs predicted -----
    errors = (y_true - y_pred).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    axes[0].hist(errors, bins=60, edgecolor='k', alpha=0.7, color='coral')
    axes[0].set_xlabel('Error', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(y_true.flatten(), y_pred.flatten(), s=8, alpha=0.25, color='navy')
    mn = min(y_true.min(), y_pred.min()); mx = max(y_true.max(), y_pred.max())
    axes[1].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Values', fontsize=11)
    axes[1].set_ylabel('Predicted Values', fontsize=11)
    axes[1].set_title('True vs Predicted', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p3 = os.path.join(savedir, 'error_and_scatter.png')
    fig.savefig(p3, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"✓ Saved: {p3}")

    # ----- Plot 4: NMSE vs Rho (THIS IS THE NEW PLOT) -----
    fig = plt.figure(figsize=(9, 5))
    # Plot the binned average
    plt.plot(bin_centers, binned_nmse, 'o-', color='purple', linewidth=2, markersize=8, label='Avg. NMSE (dB) per bin')
    # Plot the raw scatter underneath
    plt.scatter(eval_df['rho'], eval_df['nmse_db'], alpha=0.05, color='grey', s=10, label='Per-sample NMSE (dB)')
    
    plt.xlabel('Correlation Coefficient (rho)', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title('Prediction Performance vs. Channel Correlation', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    p4 = os.path.join(savedir, 'nmse_vs_rho.png')
    fig.savefig(p4, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"✓ Saved: {p4}")

    return y_pred, y_true, y_pred_c, y_true_c, metrics


# ------------------------------
# Utility: plot training history (saved)
# ------------------------------
def plot_training_history(history, savedir='.'):
    """
    Plot and save training history.
    """
    fig = plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History - Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2, color='blue')
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Training History - MAE', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(savedir, 'training_history.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {p}")


# ------------------------------
# MAIN
# ------------------------------
def main():
    """
    Main function: Complete ML pipeline from loading data to evaluation.
    """
    print("\n" + "="*60)
    print(" PART 2: MODEL TRAINING & EVALUATION")
    print(" Load Dataset → Train → Evaluate")
    print("="*60)
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    DATASET_DIR = './dataset'    # Directory where CSV files are stored
    PREFIX = 'channel_data'      # Prefix of CSV files
    N_ANTENNAS = 4               # Number of antennas (must match Part 1)
    EPOCHS = 100                 # Maximum training epochs
    BATCH_SIZE = 64              # Batch size for training
    SAVEDIR = '.'                # Directory to save outputs
    
    print("\nConfiguration:")
    print(f"  Dataset directory: {DATASET_DIR}")
    print(f"  File prefix: {PREFIX}")
    print(f"  Number of antennas: {N_ANTENNAS}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Step 1: Load dataset from CSV
    X, y, rho_values = load_dataset_from_csv(input_dir=DATASET_DIR, prefix=PREFIX)

    # Step 1.5: Plot the data physics (NEW)
    plot_data_physics(X, y, rho_values, n_antennas=N_ANTENNAS, savedir=SAVEDIR)

    # Step 2: Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, rho_train, rho_val, rho_test, scaler_X, scaler_y = preprocess_data(
        X, y, rho_values
    )

    # Step 3: Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, features)
    output_dim = y_train.shape[1]
    model = build_lstm_model(input_shape, output_dim, lstm_units=64, dense_units=32, dropout=0.2)

    # Step 4: Train model
    history = train_model(model, X_train, y_train, X_val, y_val,
                          epochs=EPOCHS, batch_size=BATCH_SIZE, savedir=SAVEDIR)

    # Step 5: Plot training history
    plot_training_history(history, savedir=SAVEDIR)

    # Step 6: Evaluate & save test plots
    y_pred, y_true, y_pred_c, y_true_c, metrics = evaluate_model_and_plot_all(
        model, X_test, y_test, rho_test, scaler_y, n_antennas=N_ANTENNAS, n_display=5, savedir=SAVEDIR
    )

    # Step 7: Save final model
    final_path = os.path.join(SAVEDIR, 'final_model.keras')
    model.save(final_path)
    print(f"\n✓ Saved final model to: {final_path}")

    print("\n" + "="*60)
    print(" PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 data_physics_h_vs_h.png - h(t+1) vs h(t) analysis (NEW)")
    print("  📊 training_history.png - Training progress")
    print("  📊 predictions_samples.png - Sample predictions")
    print("  📊 per_antenna_nmse.png - Per-antenna performance")
    print("  📊 error_and_scatter.png - Error analysis")
    print("  📊 nmse_vs_rho.png - Performance vs. Rho (NEW)")
    print("  🤖 best_model.keras - Best model (during training)")
    print("  🤖 final_model.keras - Final trained model")
    print("\nTo use the trained model later:")
    print("  >>> from tensorflow import keras")
    print("  >>> model = keras.models.load_model('final_model.keras')")
    print("  >>> predictions = model.predict(new_data)")


if __name__ == "__main__":
    main()