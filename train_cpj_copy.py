"""
CosmoPower-JAX Training Script

Author: A. Spurio Mancini
"""

import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.nn import sigmoid
import optax
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import time
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX


@dataclass
class TrainingConfig:
    """Configuration for training CosmoPower-JAX models."""
    n_hidden: int = 512
    n_layers: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 1000
    early_stopping_patience: int = 50
    validation_split: float = 0.1
    log_spectra: bool = True  # Whether to train on log-transformed spectra
    use_pca: bool = False     # Whether to use PCA preprocessing
    n_pca_components: Optional[int] = None
    architecture: str = "standard"
    random_seed: int = 42
    use_lr_scheduler: bool = True  # Whether to use learning rate scheduler
    lr_decay_factor: float = 0.95  # Factor to multiply LR by when decaying
    lr_decay_patience: int = 10    # Epochs to wait before decaying LR


class CosmoPowerJAXTrainer:
    """Trainer for CosmoPower-JAX neural networks."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.key = random.PRNGKey(config.random_seed)

        # Initialize normalization parameters
        self.param_train_mean = None
        self.param_train_std = None
        self.feature_train_mean = None
        self.feature_train_std = None
        self.pca_matrix = None
        self.training_mean = None
        self.training_std = None

    def custom_activation(self, x: jnp.ndarray, alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        """
        Custom activation function
        """
        return jnp.multiply(
            jnp.add(beta, jnp.multiply(sigmoid(jnp.multiply(alpha, x)), jnp.subtract(1., beta))),
            x
        )

    def init_network(self, n_params: int, n_features: int) -> Tuple[List[Tuple], List[Tuple]]:
        """Initialize network weights and hyperparameters."""
        weights = []
        hyper_params = []

        # Input layer
        self.key, subkey = random.split(self.key)
        w = random.normal(subkey, (n_params, self.config.n_hidden)) * 0.1
        b = jnp.zeros((self.config.n_hidden,))
        weights.append((w, b))

        # Hyperparameters for activation
        self.key, subkey1, subkey2 = random.split(self.key, 3)
        alpha = random.normal(subkey1, (self.config.n_hidden,)) * 0.1
        beta = random.uniform(subkey2, (self.config.n_hidden,), minval=0.0, maxval=1.0)
        hyper_params.append((alpha, beta))

        # Hidden layers
        for i in range(self.config.n_layers - 2):
            self.key, subkey = random.split(self.key)
            w = random.normal(subkey, (self.config.n_hidden, self.config.n_hidden)) * 0.1
            b = jnp.zeros((self.config.n_hidden,))
            weights.append((w, b))

            self.key, subkey1, subkey2 = random.split(self.key, 3)
            alpha = random.normal(subkey1, (self.config.n_hidden,)) * 0.1
            beta = random.uniform(subkey2, (self.config.n_hidden,), minval=0.0, maxval=1.0)
            hyper_params.append((alpha, beta))

        # Output layer
        self.key, subkey = random.split(self.key)
        w = random.normal(subkey, (self.config.n_hidden, n_features)) * 0.1
        b = jnp.zeros((n_features,))
        weights.append((w, b))

        return weights, hyper_params

    def forward_pass(self, weights: List[Tuple], hyper_params: List[Tuple],
                    x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the network."""
        # Standardize input
        x_norm = (x - self.param_train_mean) / self.param_train_std

        # Hidden layers with custom activation
        for i in range(len(weights) - 1):
            w, b = weights[i]
            alpha, beta = hyper_params[i]

            x_norm = jnp.dot(x_norm, w) + b
            x_norm = self.custom_activation(x_norm, alpha, beta)

        # Output layer (no activation)
        w_out, b_out = weights[-1]
        output = jnp.dot(x_norm, w_out) + b_out

        # Denormalize output
        output = output * self.feature_train_std + self.feature_train_mean

        # Apply final transformations based on model type
        if self.config.log_spectra:
            output = 10**output
        elif self.config.use_pca and self.pca_matrix is not None:
            # Apply inverse PCA transformation
            output = jnp.dot(output, self.pca_matrix) * self.training_std + self.training_mean

        return output

    def loss_function(self, weights: List[Tuple], hyper_params: List[Tuple],
                     x_batch: jnp.ndarray, y_batch: jnp.ndarray) -> jnp.ndarray:
        """Compute MSE loss."""
        predictions = vmap(self.forward_pass, in_axes=(None, None, 0))(weights, hyper_params, x_batch)
        return jnp.mean((predictions - y_batch)**2)

    def prepare_data(self, params: np.ndarray, spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and normalize training data."""
        # Store parameter normalization
        self.param_train_mean = jnp.array(np.mean(params, axis=0))
        self.param_train_std = jnp.array(np.std(params, axis=0))

        # Handle spectra preprocessing
        if self.config.log_spectra:
            # Take log of spectra for training
            processed_spectra = np.log10(spectra + 1e-30)  # Add small value to avoid log(0)
        else:
            processed_spectra = spectra.copy()

        # Apply PCA if requested
        if self.config.use_pca:
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn")

            # Store original spectra statistics for inverse transform
            self.training_mean = jnp.array(np.mean(spectra, axis=0))
            self.training_std = jnp.array(np.std(spectra, axis=0))

            # Normalize spectra before PCA
            normalized_spectra = (spectra - self.training_mean) / self.training_std

            # Compute PCA
            n_components = self.config.n_pca_components or min(spectra.shape) // 2
            pca = PCA(n_components=n_components)
            processed_spectra = pca.fit_transform(normalized_spectra)
            self.pca_matrix = jnp.array(pca.components_.T)  # Shape: (n_features, n_components)

        # Store feature normalization
        self.feature_train_mean = jnp.array(np.mean(processed_spectra, axis=0))
        self.feature_train_std = jnp.array(np.std(processed_spectra, axis=0))

        return jnp.array(params), jnp.array(processed_spectra)

    def create_batches(self, x: jnp.ndarray, y: jnp.ndarray, batch_size: int):
        """Create random batches for training."""
        n_samples = x.shape[0]
        indices = jnp.arange(n_samples)

        # Shuffle indices
        self.key, subkey = random.split(self.key)
        shuffled_indices = random.permutation(subkey, indices)

        # Create batches
        for i in range(0, n_samples, batch_size):
            batch_indices = shuffled_indices[i:i+batch_size]
            yield x[batch_indices], y[batch_indices]

    def train(self, params: np.ndarray, spectra: np.ndarray,
              validation_params: Optional[np.ndarray] = None,
              validation_spectra: Optional[np.ndarray] = None,
              current_time: str = "", 
              run_name: str = "default") -> Tuple[List[Tuple], List[Tuple]]:
        """Train the neural network."""
        print("Preparing data...")
        x_train, y_train = self.prepare_data(params, spectra)

        # Split validation data if not provided
        if validation_params is None:
            n_val = int(len(x_train) * self.config.validation_split)
            self.key, subkey = random.split(self.key)
            val_indices = random.choice(subkey, len(x_train), (n_val,), replace=False)
            train_indices = jnp.setdiff1d(jnp.arange(len(x_train)), val_indices)

            x_val, y_val = x_train[val_indices], y_train[val_indices]
            x_train, y_train = x_train[train_indices], y_train[train_indices]
        else:
            x_val, y_val = self.prepare_data(validation_params, validation_spectra)

        print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

        # Initialize network
        n_params = x_train.shape[1]
        n_features = y_train.shape[1]
        weights, hyper_params = self.init_network(n_params, n_features)

        # Initialize optimizer
        optimizer = optax.adam(self.config.learning_rate)
        opt_state = optimizer.init((weights, hyper_params))

        # JIT compile loss function
        loss_fn = jit(self.loss_function)

        def create_update_step(opt):
            @jit
            def update_step(weights, hyper_params, opt_state, x_batch, y_batch):
                loss, grads = jax.value_and_grad(self.loss_function, argnums=(0, 1))(
                    weights, hyper_params, x_batch, y_batch
                )
                updates, opt_state = opt.update(grads, opt_state)
                weights, hyper_params = optax.apply_updates((weights, hyper_params), updates)
                return weights, hyper_params, opt_state, loss
            return update_step

        update_step = create_update_step(optimizer)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        lr_patience_counter = 0
        current_lr = self.config.learning_rate
        
        # Initialize best model parameters
        best_weights = weights
        best_hyper_params = hyper_params
        
        # Lists to store loss history for plotting
        train_losses = []
        val_losses = []
        learning_rates = []
        epochs_completed = []

        print("Starting training...")
        for epoch in tqdm(range(self.config.n_epochs), desc="Training"):
            # Training
            epoch_losses = []
            for x_batch, y_batch in self.create_batches(x_train, y_train, self.config.batch_size):
                weights, hyper_params, opt_state, batch_loss = update_step(
                    weights, hyper_params, opt_state, x_batch, y_batch
                )
                epoch_losses.append(batch_loss)

            train_loss = jnp.mean(jnp.array(epoch_losses))

            # Validation
            val_loss = loss_fn(weights, hyper_params, x_val, y_val)
            
            # Learning rate scheduling
            if self.config.use_lr_scheduler:
                if val_loss < best_val_loss:
                    lr_patience_counter = 0
                else:
                    lr_patience_counter += 1
                    
                # Decay learning rate if no improvement for lr_decay_patience epochs
                if lr_patience_counter >= self.config.lr_decay_patience:
                    current_lr *= self.config.lr_decay_factor
                    # Update optimizer with new learning rate and recreate update function
                    optimizer = optax.adam(current_lr)
                    opt_state = optimizer.init((weights, hyper_params))
                    update_step = create_update_step(optimizer)
                    lr_patience_counter = 0
                    print(f"Learning rate decayed to {current_lr:.2e} at epoch {epoch}")
            
            # Store losses and learning rate for plotting
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            learning_rates.append(current_lr)
            epochs_completed.append(epoch)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = weights
                best_hyper_params = hyper_params
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        # Create and save loss evolution plot
        self.plot_training_history(epochs_completed, train_losses, val_losses, learning_rates, current_time, run_name)
        
        return best_weights, best_hyper_params

    def plot_training_history(self, epochs: List[int], train_losses: List[float], val_losses: List[float], learning_rates: List[float], current_time: str, run_name: str):
        """Plot training and validation loss evolution along with learning rate."""
        plt.figure(figsize=(18, 5))
        
        # Plot 1: Linear scale
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Log scale
        plt.subplot(1, 3, 2)
        plt.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training and Validation Loss Evolution (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate Evolution
        plt.subplot(1, 3, 3)
        plt.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Use scientific notation for y-axis if learning rates are small
        if max(learning_rates) < 0.01:
            plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save the plot with timestamped filename in plots directory
        plot_filename = f'{current_time}_{run_name}_training_loss_evolution.png'
        plot_filepath = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved as: {plot_filepath}")
        
        # Show the plot
        plt.show()

    def save_model(self, weights: List[Tuple], hyper_params: List[Tuple],
                   filename: str, probe: str, parameters: List[str],
                   modes: np.ndarray):
        """Save model in CosmoPowerJAX-compatible format."""

        # Convert JAX arrays to numpy and separate weights/biases and alphas/betas
        weights_ = [np.array(w) for w, b in weights]  # Don't transpose - CPJ will do it
        biases_ = [np.array(b) for w, b in weights]
        alphas_ = [np.array(a) for a, b in hyper_params]
        betas_ = [np.array(b) for a, b in hyper_params]

        # Create the model tuple in the expected format for custom_log
        # Based on lines 215-220 in cosmopower_jax.py
        model_data = (
            weights_,                                # weights_ (list of weight matrices)
            biases_,                                 # biases_ (list of bias vectors)
            alphas_,                                 # alphas_ (list of alpha parameters)
            betas_,                                  # betas_ (list of beta parameters)
            np.array(self.param_train_mean),         # param_train_mean
            np.array(self.param_train_std),          # param_train_std
            np.array(self.feature_train_mean),       # feature_train_mean
            np.array(self.feature_train_std),        # feature_train_std
            len(parameters),                         # n_parameters
            parameters,                              # parameters
            len(modes),                              # n_modes
            modes,                                   # modes
            self.config.n_hidden,                    # n_hidden
            self.config.n_layers,                    # n_layers
            self.config.architecture                 # architecture
        )

        # Save the model as pickle
        dirname = os.path.dirname(filename)
        if dirname:  # Only create directory if filename includes a path
            os.makedirs(dirname, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filename}")

        # Also save as .npz format for compatibility with TF>=2.14 fallback
        npz_filename = filename.replace('.pkl', '.npz')
        self.save_model_npz(weights_, biases_, alphas_, betas_, npz_filename, parameters, modes)
        print(f"Model also saved to {npz_filename} for TF>=2.14 compatibility")

    def save_model_npz(self, weights_: List, biases_: List, alphas_: List, betas_: List,
                       filename: str, parameters: List[str], modes: np.ndarray):
        """Save model in .npz dictionary format for TF>=2.14 compatibility."""

        # Create dictionary with all model data in the format expected by CPJ
        model_dict = {
            'weights_': weights_,
            'biases_': biases_,
            'alphas_': alphas_,
            'betas_': betas_,
            'param_train_mean': np.array(self.param_train_mean),
            'param_train_std': np.array(self.param_train_std),
            'feature_train_mean': np.array(self.feature_train_mean),
            'feature_train_std': np.array(self.feature_train_std),
            'n_parameters': len(parameters),
            'parameters': parameters,
            'n_modes': len(modes),
            'modes': modes,
            'n_hidden': self.config.n_hidden,
            'n_layers': self.config.n_layers,
            'architecture': self.config.architecture
        }

        # Add PCA data if used
        if self.config.use_pca:
            model_dict.update({
                'training_mean': np.array(self.training_mean),
                'training_std': np.array(self.training_std),
                'pca_matrix': np.array(self.pca_matrix)
            })

        # Save as .npz with the dictionary as arr_0 (CPJ expects this format)
        np.savez(filename, arr_0=model_dict)


def generate_dummy_data(probe: str, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Generate dummy training data for testing."""

    if probe in ['cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp']:
        # CMB parameters: omega_b, omega_cdm, h, tau, n_s, ln10^10A_s
        parameters = ['omega_b', 'omega_cdm', 'h', 'tau_reio', 'n_s', 'ln10^{10}A_s']
        params = np.array([
            np.random.uniform(0.019, 0.025, n_samples),  # omega_b
            np.random.uniform(0.10, 0.14, n_samples),   # omega_cdm
            np.random.uniform(0.64, 0.74, n_samples),   # h
            np.random.uniform(0.04, 0.12, n_samples),   # tau
            np.random.uniform(0.92, 1.00, n_samples),   # n_s
            np.random.uniform(2.9, 3.3, n_samples)      # ln10^10A_s
        ]).T

        # CMB modes (ell)
        modes = np.arange(2, 2509)
        n_modes = len(modes)

        # Generate dummy spectra with realistic CMB-like shape
        spectra = np.zeros((n_samples, n_modes))
        for i in range(n_samples):
            # Simple CMB-like power spectrum
            ell = modes.astype(float)
            As = 10**(params[i, 5] - 10)  # A_s
            ns = params[i, 4]             # n_s

            # Simplified CMB spectrum
            spectra[i] = As * (ell / 100)**(ns - 1) * np.exp(-ell / 1000) * 1e12

    elif probe in ['mpk_lin', 'mpk_boost']:
        # Matter power spectrum parameters
        parameters = ['omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s', 'z']
        params = np.array([
            np.random.uniform(0.019, 0.025, n_samples),  # omega_b
            np.random.uniform(0.10, 0.14, n_samples),   # omega_cdm
            np.random.uniform(0.64, 0.74, n_samples),   # h
            np.random.uniform(0.92, 1.00, n_samples),   # n_s
            np.random.uniform(2.9, 3.3, n_samples),     # ln10^10A_s
            np.random.uniform(0.0, 3.0, n_samples)      # z
        ]).T

        # k modes
        modes = np.logspace(-4, 2, 420)  # k in h/Mpc
        n_modes = len(modes)

        # Generate dummy matter power spectra
        spectra = np.zeros((n_samples, n_modes))
        for i in range(n_samples):
            k = modes
            As = 10**(params[i, 4] - 10)  # A_s
            ns = params[i, 3]             # n_s
            z = params[i, 5]              # redshift

            # Simple matter power spectrum
            if probe == 'mpk_lin':
                spectra[i] = As * (k / 0.05)**(ns - 1) * (1 + z)**(-2) * 1e4
            else:  # mpk_boost
                spectra[i] = 1 + 0.1 * k * (1 + z)  # Simple boost factor

    else:
        raise ValueError(f"Unknown probe: {probe}")

    return params, spectra, parameters, modes

def main():
    """Main function to train and test a CosmoPower-JAX model."""

    # Configuration
    probe = 'mpk_lin'  # Can be 'cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp', 'mpk_lin', 'mpk_boost', etc.
    model_filename = 'trained_cp_jax_model.pkl'
    n_samples = 5000  # Number of training samples
    n_test = 100      # Number of test samples
    
    # Generate timestamp and run name for file naming
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"{probe}_demo_run"  # You can customize this run name as needed

    print(f"=== CosmoPower-JAX Training and Testing Script ===")
    print(f"Probe: {probe}")
    print(f"Training samples: {n_samples}")
    print(f"Test samples: {n_test}")
    print(f"Run name: {run_name}")
    print(f"Timestamp: {current_time}")
    print()

    # ========================================
    # STEP 1: TRAINING
    # ========================================
    print("STEP 1: Training a new CosmoPower-JAX model")
    print("-" * 50)

    # Create training configuration
    config = TrainingConfig(
        n_hidden=512,           # Smaller network for faster training
        n_layers=4,
        learning_rate=1e-2,
        batch_size=512,
        n_epochs=1000,           # Fewer epochs for demonstration
        early_stopping_patience=100,
        validation_split=0.15,
        log_spectra=(probe in ['cmb_tt', 'cmb_ee', 'mpk_lin', 'mpk_boost']),
        use_pca=(probe in ['cmb_te', 'cmb_pp']),
        random_seed=42,
        use_lr_scheduler=True,  # Enable learning rate scheduling
        lr_decay_factor=0.1,   # Decay factor for learning rate
        lr_decay_patience=40    # Epochs to wait before decaying LR
    )

    # Initialize trainer
    trainer = CosmoPowerJAXTrainer(config)

    # Generate training data
    print("Generating training data...")
    params, spectra, parameters, modes = generate_dummy_data(probe, n_samples)
    print(f"Parameter shape: {params.shape}")
    print(f"Spectra shape: {spectra.shape}")
    print(f"Parameters: {parameters}")
    print()

    # Train the model
    print("Training the model...")
    start_time = time.time()
    weights, hyper_params = trainer.train(params, spectra, current_time=current_time, run_name=run_name)
    end_time = time.time()
    
    training_duration = end_time - start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = training_duration % 60
    
    print(f"Training completed in: {hours:02d}h {minutes:02d}m {seconds:05.2f}s")
    print(f"Total training time: {training_duration:.2f} seconds")
    print()

    # Save the trained model
    trainer.save_model(weights, hyper_params, model_filename, probe, parameters, modes)
    print()

    # ========================================
    # STEP 2: TESTING WITH PRE-TRAINED MODEL
    # ========================================
    print("STEP 2: Testing with pre-trained models")
    print("-" * 50)

    # Test with a standard pre-trained model (if available)
    try:
        print(f"Loading pre-trained {probe} model...")
        cp_pretrained = CosmoPowerJAX(probe=probe)
        print(f"Successfully loaded pre-trained {probe} model")

        # Generate test parameters
        test_params, _, _, _ = generate_dummy_data(probe, n_test)

        # Make predictions with pre-trained model
        print("Making predictions with pre-trained model...")
        predictions_pretrained = cp_pretrained.predict(test_params)
        print(f"Pre-trained predictions shape: {predictions_pretrained.shape}")
        print(f"Pre-trained prediction range: [{np.min(predictions_pretrained):.2e}, {np.max(predictions_pretrained):.2e}]")
        print()

    except Exception as e:
        print(f"Could not load pre-trained model: {e}")
        print("This might be expected if the pre-trained model is not available.")
        print()

    # ========================================
    # STEP 3: TESTING WITH NEWLY TRAINED MODEL
    # ========================================
    print("STEP 3: Testing with newly trained model")
    print("-" * 50)

    # Load the model we just trained
    print(f"Loading newly trained model from {model_filename}...")
    cp_trained = CosmoPowerJAX(probe='custom_log', filepath=model_filename)
    print("Successfully loaded newly trained model")

    # Generate test parameters
    test_params, test_spectra_true, _, _ = generate_dummy_data(probe, n_test)

    # Make predictions with trained model
    print("Making predictions with newly trained model...")
    predictions_trained = cp_trained.predict(test_params)
    print(f"Trained predictions shape: {predictions_trained.shape}")
    print(f"Trained prediction range: [{np.min(predictions_trained):.2e}, {np.max(predictions_trained):.2e}]")

    # ========================================
    # STEP 4: COMPUTE DERIVATIVES (BONUS)
    # ========================================
    print()
    print("STEP 4: Computing derivatives")
    print("-" * 50)

    # Test derivative computation
    test_single_param = test_params[0:1]  # Take first test sample
    print("Computing derivatives for a single parameter set...")

    try:
        derivatives = cp_trained.derivative(test_single_param, mode='forward')
        print(f"Derivatives shape: {derivatives.shape}")
        print(f"Derivative range: [{np.min(derivatives):.2e}, {np.max(derivatives):.2e}]")
    except Exception as e:
        print(f"Could not compute derivatives: {e}")

    # ========================================
    # STEP 5: EVALUATION METRICS
    # ========================================
    print()
    print("STEP 5: Evaluation metrics")
    print("-" * 50)

    # Compare predictions with true values (for dummy data)
    mse = np.mean((predictions_trained - test_spectra_true)**2)
    rmse = np.sqrt(mse)
    relative_error = np.mean(np.abs((predictions_trained - test_spectra_true) / test_spectra_true))

    print(f"Mean Squared Error: {mse:.2e}")
    print(f"Root Mean Squared Error: {rmse:.2e}")
    print(f"Mean Relative Error: {relative_error:.2%}")

    # Create comparison plot of true vs predicted power spectra
    plt.figure(figsize=(15, 10))
    
    # Plot a few sample spectra for comparison
    n_samples_to_plot = min(5, n_test)
    
    for i in range(n_samples_to_plot):
        plt.subplot(2, 3, i + 1)
        
        if probe in ['cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp']:
            # For CMB: x-axis is ell (multipole), log-log plot
            plt.loglog(modes, test_spectra_true[i], 'b-', label='True', linewidth=2)
            plt.loglog(modes, predictions_trained[i], 'r--', label='Predicted', linewidth=2)
            plt.xlabel('Multipole ℓ')
            plt.ylabel('Power Spectrum')
            plt.title(f'CMB {probe.upper()} - Sample {i+1}')
        else:
            # For matter power spectrum: x-axis is k (wavenumber), log-log plot
            plt.loglog(modes, test_spectra_true[i], 'b-', label='True', linewidth=2)
            plt.loglog(modes, predictions_trained[i], 'r--', label='Predicted', linewidth=2)
            plt.xlabel('k [h/Mpc]')
            plt.ylabel('Power Spectrum')
            plt.title(f'Matter P(k) {probe.upper()} - Sample {i+1}')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Add a residual plot
    plt.subplot(2, 3, 6)
    # Plot residuals for the first sample
    residuals = (predictions_trained[0] - test_spectra_true[0]) / test_spectra_true[0] * 100
    
    if probe in ['cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp']:
        plt.semilogx(modes, residuals, 'g-', linewidth=2)
        plt.xlabel('Multipole ℓ')
    else:
        plt.semilogx(modes, residuals, 'g-', linewidth=2)
        plt.xlabel('k [h/Mpc]')
    
    plt.ylabel('Relative Error [%]')
    plt.title('Relative Error (Sample 1)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save the comparison plot with timestamped filename in plots directory
    comparison_plot_filename = f'{current_time}_{run_name}_power_spectra_comparison.png'
    comparison_plot_filepath = os.path.join(plots_dir, comparison_plot_filename)
    plt.savefig(comparison_plot_filepath, dpi=300, bbox_inches='tight')
    print(f"Power spectra comparison plot saved as: {comparison_plot_filepath}")
    plt.show()

    # Print some sample predictions vs truth
    print()
    print("Sample predictions vs truth (first 3 test samples, first 5 modes):")
    for i in range(min(3, n_test)):
        print(f"Sample {i+1}:")
        print(f"  True:      {test_spectra_true[i, :5]}")
        print(f"  Predicted: {predictions_trained[i, :5]}")
        print(f"  Rel Error: {np.abs((predictions_trained[i, :5] - test_spectra_true[i, :5]) / test_spectra_true[i, :5])}")
        print()

    # ========================================
    # STEP 6: PARAMETER SPACE EXPLORATION
    # ========================================
    print("STEP 6: Parameter space exploration")
    print("-" * 50)

    # Test with parameter dictionary input
    if probe in ['cmb_tt', 'cmb_ee', 'cmb_te', 'cmb_pp']:
        test_dict = {
            'omega_b': 0.022383,
            'omega_cdm': 0.12011,
            'h': 0.67556,
            'tau_reio': 0.0544,
            'n_s': 0.96605,
            'ln10^{10}A_s': 3.0448
        }
    else:  # matter power spectrum
        test_dict = {
            'omega_b': 0.022383,
            'omega_cdm': 0.12011,
            'h': 0.67556,
            'n_s': 0.96605,
            'ln10^{10}A_s': 3.0448,
            'z': 0.0
        }

    print("Testing with parameter dictionary input:")
    print(f"Parameters: {test_dict}")

    try:
        pred_dict = cp_trained.predict(test_dict)
        print(f"Prediction shape: {pred_dict.shape}")
        print(f"Prediction range: [{np.min(pred_dict):.2e}, {np.max(pred_dict):.2e}]")
        print(f"First 5 values: {pred_dict[:5]}")
    except Exception as e:
        print(f"Dictionary input failed: {e}")
        # Try with array input instead
        try:
            test_array = np.array([[test_dict[param] for param in parameters]])
            pred_array = cp_trained.predict(test_array)
            print(f"Array input successful - Prediction shape: {pred_array.shape}")
            print(f"First 5 values: {pred_array[:5]}")
        except Exception as e2:
            print(f"Array input also failed: {e2}")

    print()
    print("=" * 60)
    print("Training and testing completed successfully!")
    print("=" * 60)

    # Cleanup
    if os.path.exists(model_filename):
        print(f"Trained model saved as: {model_filename}")
    if os.path.exists(model_filename.replace('.pkl', '.npz')):
        print(f"Compatibility version saved as: {model_filename.replace('.pkl', '.npz')}")

if __name__ == "__main__":
    main()
