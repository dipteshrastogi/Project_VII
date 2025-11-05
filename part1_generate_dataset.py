#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 2025
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ------------------------------
# Data Generation Function
# ------------------------------
def generate_channel_data(n_samples=5000, n_antennas=4, rho_range=(0.6, 0.95),
                          sequence_length=10, noise_var=0.05, verbose=True):
    """
    Generate synthetic channel sequences using Gauss–Markov AR(1) model:
        h(t+1) = rho * h(t) + innovation

    Returns:
      X: shape (n_samples, sequence_length, n_features) where n_features = 2*n_antennas + 1 (rho included)
      y: shape (n_samples, 2*n_antennas)  (real parts then imag parts of h(T))
      rho_values: shape (n_samples,)
    """
    if verbose:
        print("="*60)
        print("GENERATING CHANNEL DATA")
        print(f"samples={n_samples}, antennas={n_antennas}, seq_len={sequence_length}, rho_range={rho_range}")
        print("="*60)

    all_X = []
    all_y = []
    all_rho = []

    for i in range(n_samples):
        # pick rho for this sample (simulate different dynamics)
        rho = np.random.uniform(rho_range[0], rho_range[1])

        # initialize complex channel (circular complex Gaussian, unit variance per element)
        h = (np.random.randn(n_antennas) + 1j * np.random.randn(n_antennas)) / np.sqrt(2)

        seq = []
        # produce sequence of length (sequence_length+1), last entry is target
        for t in range(sequence_length + 1):
            innovation_var = noise_var * (1 - rho**2)
            innovation = np.sqrt(max(innovation_var, 0.0)) * (
                np.random.randn(n_antennas) + 1j * np.random.randn(n_antennas)
            ) / np.sqrt(2)
            seq.append(h.copy())
            h = rho * h + innovation

        input_seq = seq[:-1]  # length = sequence_length
        target = seq[-1]      # h(T)

        # convert each time step to real features: [real parts..., imag parts..., rho]
        features = []
        for h_t in input_seq:
            h_real = h_t.real
            h_imag = h_t.imag
            features.append(np.concatenate([h_real, h_imag, [rho]]))

        all_X.append(features)
        all_y.append(np.concatenate([target.real, target.imag]))
        all_rho.append(rho)

        # progress print
        if verbose and (i+1) % 1000 == 0:
            print(f"  Generated {i+1}/{n_samples} sequences...")

    X = np.array(all_X, dtype=np.float32)  # (n_samples, seq_len, features)
    y = np.array(all_y, dtype=np.float32)  # (n_samples, 2*n_antennas)
    rho_values = np.array(all_rho, dtype=np.float32)

    if verbose:
        print("Data generation complete.")
        print("  X shape:", X.shape)
        print("  y shape:", y.shape)

    return X, y, rho_values


# ------------------------------
# Save to CSV
# ------------------------------
def save_dataset_to_csv(X, y, rho_values, output_dir='./dataset', prefix='channel_data'):
    """
    Save the generated dataset to CSV files.
    
    Creates 3 CSV files:
    1. {prefix}_X.csv - Input sequences (flattened)
    2. {prefix}_y.csv - Target outputs
    3. {prefix}_metadata.csv - Contains rho values and shape information
    
    Parameters:
    -----------
    X : numpy array
        Input features (n_samples, sequence_length, n_features)
    y : numpy array
        Target outputs (n_samples, output_dim)
    rho_values : numpy array
        Correlation coefficients (n_samples,)
    output_dir : str
        Directory to save CSV files
    prefix : str
        Prefix for filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING DATASET TO CSV")
    print("="*60)
    
    n_samples, seq_len, n_features = X.shape
    
    # Save X (input sequences) - flatten the sequence dimension
    # Each row: sample, all time steps concatenated
    X_flattened = X.reshape(n_samples, -1)  # (n_samples, seq_len * n_features)
    
    # Create column names for X
    X_columns = []
    for t in range(seq_len):
        for f in range(n_features):
            X_columns.append(f"t{t}_f{f}")
    
    X_df = pd.DataFrame(X_flattened, columns=X_columns)
    X_path = os.path.join(output_dir, f'{prefix}_X.csv')
    X_df.to_csv(X_path, index=False)
    print(f"✓ Saved input sequences to: {X_path}")
    print(f"  Shape: {X.shape} → Flattened to {X_flattened.shape}")
    
    # Save y (target outputs)
    y_columns = [f"output_{i}" for i in range(y.shape[1])]
    y_df = pd.DataFrame(y, columns=y_columns)
    y_path = os.path.join(output_dir, f'{prefix}_y.csv')
    y_df.to_csv(y_path, index=False)
    print(f"✓ Saved target outputs to: {y_path}")
    print(f"  Shape: {y.shape}")
    
    # Save metadata (rho values and shape information)
    metadata_df = pd.DataFrame({
        'rho': rho_values,
        'sample_id': np.arange(n_samples)
    })
    metadata_path = os.path.join(output_dir, f'{prefix}_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata to: {metadata_path}")
    
    # Save shape information as a separate small file
    shape_info = {
        'n_samples': [n_samples],
        'sequence_length': [seq_len],
        'n_features': [n_features],
        'output_dim': [y.shape[1]]
    }
    shape_df = pd.DataFrame(shape_info)
    shape_path = os.path.join(output_dir, f'{prefix}_shape_info.csv')
    shape_df.to_csv(shape_path, index=False)
    print(f"✓ Saved shape information to: {shape_path}")
    
    print("\n" + "="*60)
    print("DATASET SAVED SUCCESSFULLY!")
    print("="*60)
    print(f"All files saved in directory: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  1. {prefix}_X.csv - Input sequences")
    print(f"  2. {prefix}_y.csv - Target outputs")
    print(f"  3. {prefix}_metadata.csv - Rho values")
    print(f"  4. {prefix}_shape_info.csv - Dataset dimensions")


# ------------------------------
# MAIN
# ------------------------------
def main():
    """
    Main function to generate and save channel dataset.
    """
    print("\n" + "="*60)
    print(" PART 1: DATASET GENERATION")
    print(" Gauss-Markov Channel Model")
    print("="*60 + "\n")
    
    # ==========================================
    # CONFIGURATION - Modify these as needed
    # ==========================================
    N_SAMPLES = 5000        # Number of channel sequences to generate
    N_ANTENNAS = 4          # Number of channel coefficients
    SEQ_LENGTH = 10         # Input sequence length
    RHO_RANGE = (0.6, 0.95) # Range of correlation coefficient
    NOISE_VAR = 0.05        # Innovation noise variance
    OUTPUT_DIR = './dataset' # Directory to save CSV files
    PREFIX = 'channel_data' # Prefix for CSV filenames
    
    print("Configuration:")
    print(f"  Number of samples: {N_SAMPLES}")
    print(f"  Number of antennas: {N_ANTENNAS}")
    print(f"  Sequence length: {SEQ_LENGTH}")
    print(f"  Rho range: {RHO_RANGE}")
    print(f"  Noise variance: {NOISE_VAR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    
    # Step 1: Generate data
    X, y, rho_values = generate_channel_data(
        n_samples=N_SAMPLES,
        n_antennas=N_ANTENNAS,
        rho_range=RHO_RANGE,
        sequence_length=SEQ_LENGTH,
        noise_var=NOISE_VAR,
        verbose=True
    )
    
    # Step 2: Save to CSV
    save_dataset_to_csv(
        X, y, rho_values,
        output_dir=OUTPUT_DIR,
        prefix=PREFIX
    )
    
    print("\n" + "="*60)
    print(" DATASET GENERATION COMPLETE!")
    print("="*60)
    print("\n✓ You can now run Part 2 (training script) to train the model.")
    print(f"✓ Dataset files are in: {OUTPUT_DIR}/")
    

if __name__ == "__main__":
    main()