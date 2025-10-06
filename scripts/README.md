## UNet-VAE RIR — Scripts

This folder contains the core training, evaluation, and utility code for the UNet-VAE Room Impulse Response (RIR) synthesis project. The code implements deep learning models (U-Net, VAE, hybrid architectures) to synthesize RIRs from room acoustic parameters.

**Authors:** Ignacio Martín, José Antonio Belloch, Gema Piñero  
**Institution:** University Carlos III de Madrid

---

## Core Scripts

### Training & Evaluation

**`main_training.py`** — Main training script for all models
- Trains deep learning models (UNet, UNetVAE, UNetVAEEmb, UNetN, Autoencoder, ResAE, VAE) on RIR datasets
- **Configuration:** Edit variables at top of `if __name__ == '__main__':` block:
  - `name` — Model architecture: `'unet-vae'`, `'unet-n'`, `'unet'`, `'autoencoder'`, `'res-ae'`, `'vae'`
  - `latent_space_dim` — Bottleneck dimension (64, 128, 256)
  - `loss` — Loss function: `'mae'` or `'mse'`
  - `diff` — Whether to use difference loss (True/False)
  - `n_epochs`, `lr`, `batch_size_per_replica` — Training hyperparameters
  - `rooms`, `arrays`, `zones` — Dataset filtering (None = all)
- **Multi-GPU support:** Uses TensorFlow `MirroredStrategy` for distributed training
- **Output:** Saves checkpoints to `../results/{model_name+modifier}/`

**`rir_generation.py`** — RIR generation and evaluation script
- Loads trained models and generates RIRs for test sets
- Computes comprehensive metrics: MSE (spectrogram, magnitude, phase, waveform), SDR, similarity, energy decay curve (EDC) analysis
- **Configuration:** Edit variables in script body (model name, checkpoint path, algorithm)
- **Output:** Generated RIRs, loss metrics CSV files, timing benchmarks per room type

### Data & Utilities

**`dataset.py`** — Dataset class for loading and managing RIR data
- **Class:** `Dataset` — Loads RIRs from zip archives, extracts STFT features (magnitude + phase), creates embeddings from room parameters
- **Parameters:**
  - `dir_dataset` — Path to dataset root
  - `dataset_name` — Dataset identifier
  - `room` — List of room types: `['HemiAnechoicRoom', 'LargeMeetingRoom', 'MediumMeetingRoom', 'ShoeBoxRoom', 'SmallMeetingRoom']` or `None` for all
  - `array` — Microphone array types: `['PlanarMicrophoneArray', 'CircularMicrophoneArray']`
  - `zone` — Zones: `['ZoneA', 'ZoneB', 'ZoneC', 'ZoneD', 'ZoneE']`
  - `normalization=True` — Apply normalization to features
  - `normalize_vector=False` — Normalize embedding vectors (required for `unet-vae`, `unet-n`)
  - `downsample=False` — Downsample audio to 16kHz

**`datageneratorv2.py`** — Keras data generator for batching
- **Class:** `DataGenerator(Sequence)` — Generates training/validation/test batches
- Splits data: 70% train / 20% validation / 10% test
- Outputs: spectrograms (magnitude + phase), embeddings, optional room characteristics

**`preprocess.py`** — Preprocessing utilities (classes, not executable)
- `FeatureExtractor` — STFT extraction (n_fft=256, win_length=128, hop_length=64)
- `Normalizer` — Log-scale magnitude normalization, phase normalization to [0, 1]
- `Loader` — Audio file loading with librosa (48kHz, 0.2s duration)
- `TensorPadder` — Pads spectrograms to fixed shape (144, 160)

**`postprocess.py`** — Postprocessing utilities (classes, not executable)
- **Class:** `PostProcess` — Inverse STFT, denormalization, waveform reconstruction
- Supports Griffin-Lim phase reconstruction (`algorithm="gl"`) or direct phase (`algorithm="ph"`)
- **Methods:** `post_process()` — Converts model output (normalized STFT) back to audio waveform

**`rooms.py`** — Room geometry classes
- **Classes:** `Quadrilateral`, `Room`, `UTSRoom`
- `UTSRoom` — Defines room geometry, microphone array positions, computes embeddings from room parameters
- Embedding format: `[a, b, c, d, alpha, beta, gamma, delta, height, xl, yl, zl, xm, ym, zm, rt60]` (16 dimensions)

**`visualize.py`** — Plotting and visualization utilities
- Functions: `plot_wav()`, `plot_spec()`, `plot_feature_vs_wav()`, `plot_feature_vs_feature_wav()`, `plot_phase_vs_phase()`
- Saves comparison plots of true vs predicted spectrograms and waveforms

---

## Model Architectures (`dl_models/`)

| File | Model | Description |
|------|-------|-------------|
| `u_net.py` | `UNet` | U-Net architecture with skip connections for RIR synthesis |
| `u_net_new.py` | `UNetN` | Modified U-Net variant with normalized embeddings |
| `unet_vae.py` | `UNetVAE` | U-Net + Variational Autoencoder (VAE) with latent space sampling |
| `unet_vae_emb.py` | `UNetVAEEmb` | UNetVAE with enhanced embedding integration |
| `autoencoder.py` | `Autoencoder` | Convolutional autoencoder baseline |
| `res_ae.py` | `ResAE` | Residual autoencoder with skip connections |
| `vae.py` | `VAE` | Standard Variational Autoencoder |
| `vqvae.py` | `VQVAE` | Vector-Quantized VAE |
| `cnn_clas.py` | — | CNN classifier utilities |
| `ae_net.py` | — | Additional autoencoder utilities |

**Common parameters:**
- `input_shape` — Spectrogram shape (144, 64, 2) or (144, 160, 2)
- `inf_vector_shape` — Embedding shape (2, 16) for [input_room, output_room] embeddings
- `latent_space_dim` — Bottleneck dimension (32, 64, 128, 256)
- `learning_rate` — Adam optimizer learning rate (typically 1e-5 to 5e-7)

---

## Directory Structure

```
scripts/
├── main_training.py          # Training entrypoint
├── rir_generation.py         # Generation & evaluation
├── dataset.py                # Dataset loader
├── datageneratorv2.py        # Batch generator
├── preprocess.py             # Preprocessing classes
├── postprocess.py            # Postprocessing classes
├── rooms.py                  # Room geometry
├── visualize.py              # Plotting utilities
├── dl_models/                # Model implementations
│   ├── u_net.py
│   ├── unet_vae.py
│   ├── autoencoder.py
│   └── ...
├── results/                  # Training checkpoints
│   └── {model_name}/
│       ├── checkpoint
│       ├── ckpt-1.data-00000-of-00001
│       └── ckpt-1.index
└── utils/                    # Compiled Python modules
```

--- 

## Quick Start

These commands use bash on Linux or macOS.

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

**Key dependencies:**
- TensorFlow 2.10+ / Keras 2.10
- librosa 0.10.1 (audio processing)
- numpy, pandas, matplotlib
- scipy, numba

### 2. Prepare Dataset

The dataset should be structured as:

```
../../datasets/room_impulse/
├── HemiAnechoicRoom/
├── LargeMeetingRoom/
├── MediumMeetingRoom/
├── ShoeBoxRoom/
└── SmallMeetingRoom/
```

Each room folder contains `.wav` files with RIR recordings. The `Dataset` class automatically extracts STFT features.

### 3. Train a Model

Edit `main_training.py` to configure your model:

```python
# Model selection
name = 'unet-vae'              # Options: 'unet-vae', 'unet-n', 'unet', 'autoencoder', 'res-ae', 'vae'
latent_space_dim = 64          # Bottleneck: 32, 64, 128, 256
loss = "mae"                   # 'mae' or 'mse'
diff = True                    # Use difference loss

# Training hyperparameters
n_epochs = 100
lr = 5e-7
batch_size_per_replica = 16

# Dataset filtering (None = all)
rooms = None                   # Or ['HemiAnechoicRoom', 'LargeMeetingRoom', ...]
arrays = ["PlanarMicrophoneArray"]
zones = None
```

Run training:

```bash
python main_training.py
```

**Multi-GPU training:** The script automatically detects available GPUs and uses `MirroredStrategy`.

**Checkpoints:** Saved to `../results/{model_name}-{latent_dim}-{loss}-{diff}/`

### 4. Generate RIRs

Edit `rir_generation.py` to configure generation:

```python
model_name = 'unet-vae'
modifier = '-64-mae-diff'
checkpoint_path = f'../results/{model_name}{modifier}/'
algorithm = 'ph'               # 'ph' (direct phase) or 'gl' (Griffin-Lim)
```

Run generation:

```bash
python rir_generation.py
```

**Output:**
- Generated RIR waveforms
- Loss metrics CSV: `{model_name}_losses.csv`
- Timing benchmarks: `{model_name}_infer_time.csv`
- Saved to: `../generated_rir/local_gen/{model_name}{modifier}/`

### 5. Visualize Results

Use `visualize.py` functions in a Python script or notebook:

```python
from visualize import plot_wav, plot_spec, plot_feature_vs_wav
import numpy as np

# Load generated waveform
waveform = np.load('path/to/generated_rir.npy')

# Plot waveform
plot_wav(waveform)

# Plot spectrogram
from preprocess import FeatureExtractor
extractor = FeatureExtractor(n_fft=256, win_length=128, hop_length=64)
amp, phase = extractor.extract(waveform)
plot_spec(amp)
```

---

## Configuration Guide

### Model Selection

| Model Name | Architecture | Normalize Vector | Use Case |
|------------|-------------|------------------|----------|
| `'unet'` | U-Net baseline | No | Fast baseline |
| `'unet-n'` | U-Net normalized | Yes | Improved embedding handling |
| `'unet-vae'` | U-Net + VAE | Yes | Best overall performance |
| `'unet-vae-emb'` | U-Net + VAE + enhanced embeddings | Yes | Complex room geometries |
| `'autoencoder'` | Standard autoencoder | No | Simple baseline |
| `'res-ae'` | Residual autoencoder | No | Deeper architecture |
| `'vae'` | Standard VAE | No | Probabilistic baseline |

### Loss Functions

- **`'mae'`** (Mean Absolute Error) — Better for preserving waveform shape
- **`'mse'`** (Mean Squared Error) — Penalizes large errors more heavily

### Difference Loss (`diff`)

When `diff=True`, the model learns to predict the *difference* between input and output RIRs rather than the output directly. Often improves results for room-to-room transfer tasks.

### Dataset Filtering

```python
# Train on specific room types
rooms = ['HemiAnechoicRoom', 'LargeMeetingRoom']

# Use only planar microphone arrays
arrays = ['PlanarMicrophoneArray']

# Train on specific zones
zones = ['ZoneA', 'ZoneB', 'ZoneC']
```

---

## Advanced Usage

### Custom Training Loop

The training script uses a custom loop (not `model.fit()`) for fine-grained control:

```python
# In main_training.py, around line 200-400
for epoch in range(n_epochs):
    train_generator.on_epoch_end()  # Shuffle
    
    for step in range(train_generator.__len__()):
        spec_in, emb, spec_out = train_generator.__getitem__(step)
        
        with tf.GradientTape() as tape:
            predictions = model(spec_in, emb, training=True)
            loss = compute_loss(spec_out, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Embedding Format

Room embeddings are 16-dimensional vectors:

```python
[a, b, c, d,                    # Room side lengths (4)
 alpha, beta, gamma, delta,     # Corner angles (4)
 xl, yl, zl,                    # Loudspeaker position (3)
 xm, ym, zm,                    # Microphone position (3)
 height,                        # Room height (1)
 rt60]                          # Reverberation time (1)
```

For models with `normalize_vector=True`, these are normalized to [0, 1] based on dataset min/max values.

### Multi-GPU Configuration

```python
# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Use GPUs 0, 1, 2

# MirroredStrategy automatically distributes batches
strategy = tf.distribute.MirroredStrategy()
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
```

---

## Evaluation Metrics

The `rir_generation.py` script computes:

1. **MSE Spectrogram** — Mean squared error on full STFT
2. **MSE Magnitude** — MSE on magnitude spectrogram only
3. **1-cos(Δφ) Phase** — Phase difference loss: `mean(1 - cos(phase_true - phase_pred))`
4. **MSE Waveform** — Time-domain MSE
5. **MSE Waveform 50ms** — MSE on first 50ms (early reflections)
6. **Misalignment** — Time-shift misalignment penalty
7. **SDR** (Signal-to-Distortion Ratio) — `10*log10(signal_power / distortion_power)`
8. **Similarity** — Cosine similarity between true and predicted waveforms
9. **Energy Similarity** — Similarity of energy decay curves (EDC)

Results are saved per room type: Global, HemiAnechoic, Large, Medium, Shoe, Small.

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM) errors**
- Reduce `batch_size_per_replica` in `main_training.py`
- Use fewer GPUs: `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`
- Enable mixed precision training (add to training script):
  ```python
  from tensorflow.keras import mixed_precision
  mixed_precision.set_global_policy('mixed_float16')
  ```

**2. Checkpoint loading errors**
- Ensure model configuration matches checkpoint:
  - Same `name`, `latent_space_dim`, `loss`, `diff` settings
  - Same `mode` parameter for UNet models
- Check TensorFlow version compatibility (script uses TF 2.10)

**3. Dataset not found**
- Verify dataset path in `main_training.py`: 
  ```python
  dataset = Dataset('../../datasets', 'room_impulse', ...)
  ```
- Ensure dataset structure matches expected format (see Quick Start section)

**4. CUDA errors**
- Check GPU availability: `nvidia-smi`
- Set specific GPUs: `export CUDA_VISIBLE_DEVICES=0,1`
- Verify CUDA/cuDNN versions match TensorFlow requirements

**5. Import errors for `dl_models`**
- Ensure you're running scripts from the `scripts/` directory
- Check Python path includes current directory:
  ```bash
  cd scripts/
  python main_training.py
  ```

**6. Phase reconstruction artifacts**
- Try different algorithms in `rir_generation.py`:
  - `algorithm='ph'` — Use predicted phase directly (faster)
  - `algorithm='gl'` — Griffin-Lim iterative phase reconstruction (slower, sometimes better)
  - Adjust `n_iters` for Griffin-Lim (default 32, try 50-100)

**7. Poor generation quality**
- Increase training epochs (`n_epochs`)
- Try different loss functions (`'mae'` vs `'mse'`)
- Enable difference loss (`diff=True`)
- Increase model capacity (`latent_space_dim`)
- Check if `normalize_vector=True` for VAE models

---

## Performance Tips

### Training Speed

1. **Use multiple GPUs**: The script supports automatic multi-GPU distribution
2. **Increase batch size**: Scale `batch_size_per_replica` with GPU memory
3. **Reduce dataset size** for debugging: Filter `rooms`, `arrays`, or `zones`
4. **Enable XLA compilation** (add to script):
   ```python
   tf.config.optimizer.set_jit(True)
   ```

### Generation Speed

1. **Use direct phase** (`algorithm='ph'`) instead of Griffin-Lim
2. **Batch generation**: `rir_generation.py` processes batches automatically
3. **GPU inference**: Ensure checkpoints are on GPU, not CPU

### Memory Optimization

1. **Downsampling**: Set `downsample=True` in Dataset (48kHz → 16kHz)
   - Changes `input_shape` to `(144, 64, 2)`
2. **Gradient checkpointing**: For very deep models (not implemented by default)
3. **Mixed precision**: Use FP16 for forward pass, FP32 for gradients

---

## File Formats

### Checkpoint Files

TensorFlow checkpoint format (v2):
```
results/{model_name}/
├── checkpoint              # Checkpoint metadata
├── ckpt-1.data-00000-of-00001  # Model weights
└── ckpt-1.index            # Weight index
```

Load checkpoint:
```python
model = UNetVAE(input_shape=(144, 64, 2), inf_vector_shape=(2, 16), ...)
model.model.load_weights('results/unet-vae-64-mae-diff/ckpt-1')
```

### Generated RIR Files

Generated RIRs are saved as:
- `.wav` files (if using `PostProcess.save_wav()`)
- `.npy` files (NumPy arrays, waveform or spectrogram)

---

## Development Workflow

### Typical Research Iteration

1. **Experiment with hyperparameters** in `main_training.py`
2. **Train model** (may take hours/days depending on dataset size)
3. **Evaluate** using `rir_generation.py` on test set
4. **Analyze metrics** from CSV files
5. **Visualize** selected samples with `visualize.py`
6. **Iterate**: Adjust architecture, loss, or data augmentation

### Adding a New Model

1. Create model class in `dl_models/new_model.py`:
   ```python
   class NewModel:
       def __init__(self, input_shape, inf_vector_shape, ...):
           # Define architecture
           self.model = self._build_model()
       
       def _build_model(self):
           # Return Keras Model
           pass
   ```

2. Import in `main_training.py`:
   ```python
   from dl_models.new_model import NewModel
   ```

3. Add to model selection:
   ```python
   if name == 'new-model':
       model = NewModel(input_shape=target_size, ...)
   ```

4. Add to `rir_generation.py` model loading section

### Debugging Tips

**Enable debug mode** in `main_training.py`:
```python
debug = True  # Loads smaller subset
```

**Check data shapes**:
```python
spec_in, emb, spec_out = train_generator.__getitem__(0)
print(f"Input: {spec_in.shape}, Embedding: {emb.shape}, Output: {spec_out.shape}")
```

**Monitor GPU usage**:
```bash
watch -n 1 nvidia-smi
```

**Profile training**:
```python
# Add TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
# Then view: tensorboard --logdir=./logs
```

---

## Citation

If you use this code in your research, please cite (pending for approval):

```bibtex
@bachelorthesis{martin2025rir,
  title={Enhanced U-Net Architectures for Accurate Room Impulse Response Generation via Differential-Phase Learning},
  author={Martin-Salinas, Ignacio and Belloch, Jose A and Piñero, Gema and Amor-Martin, Adrian},
  school={University Carlos III de Madrid},
  year={2025}
}
```

---

## License & Contact

See the repository root for license information. For questions:
- Open an issue in the project repository
- Review source code comments and docstrings
- Check TensorFlow/Keras documentation for model architecture details

---

