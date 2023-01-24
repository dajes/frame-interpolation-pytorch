# Frame interpolation in PyTorch

This is an unofficial PyTorch inference implementation
of [FILM: Frame Interpolation for Large Motion, In ECCV 2022](https://film-net.github.io/).\
[Original repository link](https://github.com/google-research/frame-interpolation)

The project is focused on creating simple and TorchScript compilable inference interface for the original pretrained TF2
model.

# Quickstart

Download a compiled model from [the release](https://github.com/dajes/frame-interpolation-pytorch/releases/tag/v1.0.0)
and specify the path to the file in the following snippet:

```python
import torch

device = torch.device('cuda')
precision = torch.float16

model = torch.jit.load(model_path, map_location='cpu')
model.eval().to(device=device, dtype=precision)

img1 = torch.rand(1, 3, 720, 1080).to(precision).to(device)
img3 = torch.rand(1, 3, 720, 1080).to(precision).to(device)
dt = img1.new_full((1, 1), .5)

with torch.no_grad():
    img2 = model(img1, img3, dt)  # Will be of the same shape as inputs (1, 3, 720, 1080)

```

# Exporting model by yourself

You will need to install TensorFlow of the version specified in
the [original repo](https://github.com/google-research/frame-interpolation#installation) and download SavedModel of "
Style" network from [there](https://github.com/google-research/frame-interpolation#pre-trained-models)

After you have downloaded the SavedModel and can load it via ```tf.compat.v2.saved_model.load(path)```:

* Clone the repository

```
git clone https://github.com/dajes/frame-interpolation-pytorch
cd frame-interpolation-pytorch
```

* Install dependencies

``` 
python -m pip install -r requirements.txt
```

* Run ```export.py```:

```
python export.py "model_path" "save_path" [--statedict] [--fp32] [--skiptest] [--gpu]
```

Argument list:

* ```model_path``` Path to the TF SavedModel
* ```save_path``` Path to save the PyTorch state dict
* ```--statedict``` Export to state dict instead of TorchScript
* ```--fp32``` Save weights at full precision
* ```--skiptest``` Skip testing and save model immediately instead
* ```--gpu``` Whether to attempt to use GPU for testing
