# Using TheNoise with Lemonade Server

TheNoise is fully integrated with [Lemonade Server](https://lemonade-server.ai) and ships as its `thenoise` backend. You can generate and edit images directly from the Lemonade web UI or through its HTTP API.

## Load a TheNoise-compatible model or a ready-made recipe

Ready-made Lemonade recipes for every supported model - checkpoints, labels
and generation defaults — are in [`recipes/`](../recipes/); each model page
links to its own recipes.

Where available, pick a model's **INT8-ConvRot** recipe over the BF16 one:
its weights are quantized to INT8 with a rotation-based scheme, so the
download and memory footprint shrink substantially and generation runs
faster, while quality holds up well.

A chosen recipe can be imported via the CLI using the command below.

```bash
lemonade import "path/to/recipe.json"
```

Alternatively, it can also be imported from the Lemonade Server web UI: download a recipe you want to try, go to
`http://localhost:13305/`, choose `File → New Model → From JSON`, select the recipe file, and wait for the model to finish downloading.

A supported model with a custom checkpoint can also be registered and pulled with a single command (example: Anima Turbo, matching [recipes/anima/Anima-Turbo.json](../recipes/anima/Anima-Turbo.json)):

```bash
lemonade pull user.Anima-Turbo \
  --checkpoint main circlestone-labs/Anima:split_files/diffusion_models/anima-turbo-v1.1.safetensors \
  --checkpoint text_encoder circlestone-labs/Anima:split_files/text_encoders/qwen_3_06b_base.safetensors \
  --checkpoint vae circlestone-labs/Anima:split_files/vae/qwen_image_vae.safetensors \
  --recipe thenoise
```

## Use the latest TheNoise version

To use the latest TheNoise version, update your Lemonade config. The `lemonade config set` command requires administrative privileges or an Admin API key because it modifies internal configuration endpoints. The key can be set explicitly via the `LEMONADE_API_KEY` environment variable before running the command.

```lemonade config set thenoise.rocm_bin="latest"```

## LoRAs

### Configure the LoRA directory

```bash
# create loras dir
sudo mkdir /var/cache/thenoise/loras

# grant permissions to lemonade
sudo chmod -R lemonade:lemonade /var/cache/thenoise/loras

# configure lora dir
lemonade config set thenoise.lora_dir="/var/cache/thenoise/loras"
```

<small>Example configuration for Arch Linux.</small>

### Use LoRAs for image generation / editing

To use a LoRA with a model:

1. Download the LoRA you want and copy its `.safetensors` file into the
   configured LoRA directory.
2. In the model's recipe `.json` file, add a `"lora_specs"` entry to the
   `"recipe_options"` block, with the LoRA name and its weight
   (several LoRAs are comma-separated): `"lora_specs": "lora-name:0.8"`.
3. Import the updated recipe in the Lemonade Server web UI at
   `http://localhost:13305/` (`File → New Model → From JSON`).
