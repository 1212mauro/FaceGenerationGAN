# Emotion-Conditioned Face Generation (Conditional WGAN-GP)

A conditional Wasserstein GAN with gradient penalty, written in PyTorch, that generates
48×48 grayscale face images showing a chosen emotion: **angry, fear, happy, sad, surprise
or neutral**.

Team project by [@1212mauro](https://github.com/1212mauro) and [@KombatKingo](https://github.com/KombatKingo).

## Model

**Generator.** A 100-dimensional noise vector is concatenated with the one-hot emotion label
and projected to a 512×12×12 feature map. Two upsample + convolution blocks bring it to 48×48,
followed by a further convolution block and a convolutional `tanh` output layer. The three
convolution blocks use **conditional batch normalization**: the emotion label sets a per-class
scale and shift, so the condition is injected at each stage rather than only at the input.

**Critic.** The label is projected to a 6×48×48 map and concatenated with the image as extra
channels. Four convolutional layers (three strided) with GroupNorm and LeakyReLU are followed
by a linear layer that outputs the Wasserstein score.

**Training.**

| Setting | Value |
|---|---|
| Loss | WGAN-GP, gradient-penalty weight λ = 10 |
| Critic updates per generator update | 10 |
| Optimizer | Adam, β = (0, 0.9) |
| Learning rate | critic 1e-4, generator 1e-5 |
| Batch size | 128 |
| Initialization | Kaiming normal for linear layers |

## Data

`preprocessor.py` expects a dataset of face images in one folder per emotion:

```
dataset/train/Angry/  Fear/  Happy/  Sad/  Surprise/  Neutral/
```

It loads every image as grayscale, resizes it to 48×48, and **balances the classes to 7,500
images each** by augmenting under-represented ones (random horizontal flip, ±10° rotation,
contrast ×0.8–1.2, brightness ±20). The data is shuffled, scaled to [-1, 1] and saved as
`train_data.npy` and `train_labels.npy` (one-hot labels). The dataset itself is not included
in this repository.

## Pretrained weights

`wass_models/` contains the generator and critic after 310 epochs
(`generator_epoch_310.pt`, `discriminator_epoch_310.pt`) and the generator and critic loss
curves for the first 300 epochs (`train_loss_gen_300.npy`, `train_loss_disc_300.npy`).

## Usage

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm

# 1. Build train_data.npy / train_labels.npy from ./dataset/train
python preprocessor.py

# 2. Load the epoch-310 weights and sample:
#    press Enter to show a 6×6 grid (each column is one emotion), type -1 to quit
python cnn_model_wass.py
```

The training script loads the `.npy` files on start-up, so run the preprocessing step first.
To train from scratch instead of loading the saved weights, set `loading = 0` in the
`__main__` block of `cnn_model_wass.py`; checkpoints are written every 10 epochs.

## Files

| File | Contents |
|---|---|
| `cnn_model_wass.py` | Generator, critic, conditional batch norm, WGAN-GP training loop, sampling |
| `preprocessor.py` | Dataset loading, class balancing with augmentation, normalization |
| `wass_models/` | Trained weights and loss curves |

## License

MIT, see [LICENSE](LICENSE).
