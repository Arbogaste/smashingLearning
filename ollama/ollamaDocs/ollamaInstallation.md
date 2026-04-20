> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Linux

## Install

To install Ollama, run the following command:

```shell  theme={"system"}
curl -fsSL https://ollama.com/install.sh | sh
```

## Manual install

<Note>
  If you are upgrading from a prior version, you should remove the old libraries
  with `sudo rm -rf /usr/lib/ollama` first.
</Note>

Download and extract the package:

```shell  theme={"system"}
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
    | sudo tar x -C /usr
```

Start Ollama:

```shell  theme={"system"}
ollama serve
```

In another terminal, verify that Ollama is running:

```shell  theme={"system"}
ollama -v
```

### AMD GPU install

If you have an AMD GPU, also download and extract the additional ROCm package:

```shell  theme={"system"}
curl -fsSL https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst \
    | sudo tar x -C /usr
```

### ARM64 install

Download and extract the ARM64-specific package:

```shell  theme={"system"}
curl -fsSL https://ollama.com/download/ollama-linux-arm64.tar.zst \
    | sudo tar x -C /usr
```

### Adding Ollama as a startup service (recommended)

Create a user and group for Ollama:

```shell  theme={"system"}
sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)
```

Create a service file in `/etc/systemd/system/ollama.service`:

```ini  theme={"system"}
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=multi-user.target
```

Then start the service:

```shell  theme={"system"}
sudo systemctl daemon-reload
sudo systemctl enable ollama
```

### Install CUDA drivers (optional)

[Download and install](https://developer.nvidia.com/cuda-downloads) CUDA.

Verify that the drivers are installed by running the following command, which should print details about your GPU:

```shell  theme={"system"}
nvidia-smi
```

### Install AMD ROCm drivers (optional)

[Download and Install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html) ROCm v6.

### Start Ollama

Start Ollama and verify it is running:

```shell  theme={"system"}
sudo systemctl start ollama
sudo systemctl status ollama
```

<Note>
  While AMD has contributed the `amdgpu` driver upstream to the official linux
  kernel source, the version is older and may not support all ROCm features. We
  recommend you install the latest driver from
  [https://www.amd.com/en/support/linux-drivers](https://www.amd.com/en/support/linux-drivers) for best support of your Radeon
  GPU.
</Note>

## Customizing

To customize the installation of Ollama, you can edit the systemd service file or the environment variables by running:

```shell  theme={"system"}
sudo systemctl edit ollama
```

Alternatively, create an override file manually in `/etc/systemd/system/ollama.service.d/override.conf`:

```ini  theme={"system"}
[Service]
Environment="OLLAMA_DEBUG=1"
```

## Updating

Update Ollama by running the install script again:

```shell  theme={"system"}
curl -fsSL https://ollama.com/install.sh | sh
```

Or by re-downloading Ollama:

```shell  theme={"system"}
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
    | sudo tar x -C /usr
```

## Installing specific versions

Use `OLLAMA_VERSION` environment variable with the install script to install a specific version of Ollama, including pre-releases. You can find the version numbers in the [releases page](https://github.com/ollama/ollama/releases).

For example:

```shell  theme={"system"}
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.5.7 sh
```

## Viewing logs

To view logs of Ollama running as a startup service, run:

```shell  theme={"system"}
journalctl -e -u ollama
```

## Uninstall

Remove the ollama service:

```shell  theme={"system"}
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service
```

Remove ollama libraries from your lib directory (either `/usr/local/lib`, `/usr/lib`, or `/lib`):

```shell  theme={"system"}
sudo rm -r $(which ollama | tr 'bin' 'lib')
```

Remove the ollama binary from your bin directory (either `/usr/local/bin`, `/usr/bin`, or `/bin`):

```shell  theme={"system"}
sudo rm $(which ollama)
```

Remove the downloaded models and Ollama service user and group:

```shell  theme={"system"}
sudo userdel ollama
sudo groupdel ollama
sudo rm -r /usr/share/ollama
```

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# macOS

## System Requirements

* MacOS Sonoma (v14) or newer
* Apple M series (CPU and GPU support) or x86 (CPU only)

## Filesystem Requirements

The preferred method of installation is to mount the `ollama.dmg` and drag-and-drop the Ollama application to the system-wide `Applications` folder.  Upon startup, the Ollama app will verify the `ollama` CLI is present in your PATH, and if not detected, will prompt for permission to create a link in `/usr/local/bin`

Once you've installed Ollama, you'll need additional space for storing the Large Language models, which can be tens to hundreds of GB in size.  If your home directory doesn't have enough space, you can change where the binaries are installed, and where the models are stored.

### Changing Install Location

To install the Ollama application somewhere other than `Applications`, place the Ollama application in the desired location, and ensure the CLI `Ollama.app/Contents/Resources/ollama` or a sym-link to the CLI can be found in your path.  Upon first start decline the "Move to Applications?" request.

## Troubleshooting

Ollama on MacOS stores files in a few different locations.

* `~/.ollama` contains models and configuration
* `~/.ollama/logs` contains logs
  * *app.log* contains most recent logs from the GUI application
  * *server.log* contains the most recent server logs
* `<install location>/Ollama.app/Contents/Resources/ollama` the CLI binary

## Uninstall

To fully remove Ollama from your system, remove the following files and folders:

```
sudo rm -rf /Applications/Ollama.app
sudo rm /usr/local/bin/ollama
rm -rf "~/Library/Application Support/Ollama"
rm -rf "~/Library/Saved Application State/com.electron.ollama.savedState"
rm -rf ~/Library/Caches/com.electron.ollama/
rm -rf ~/Library/Caches/ollama
rm -rf ~/Library/WebKit/com.electron.ollama
rm -rf ~/.ollama
```

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Windows... if you really must 

Ollama runs as a native Windows application, including NVIDIA and AMD Radeon GPU support.
After installing Ollama for Windows, Ollama will run in the background and
the `ollama` command line is available in `cmd`, `powershell` or your favorite
terminal application. As usual the Ollama [API](/api) will be served on
`http://localhost:11434`.

## System Requirements

* Windows 10 22H2 or newer, Home or Pro
* NVIDIA 452.39 or newer Drivers if you have an NVIDIA card
* AMD Radeon Driver [https://www.amd.com/en/support](https://www.amd.com/en/support) if you have a Radeon card

Ollama uses unicode characters for progress indication, which may render as unknown squares in some older terminal fonts in Windows 10. If you see this, try changing your terminal font settings.

## Filesystem Requirements

The Ollama install does not require Administrator, and installs in your home directory by default. You'll need at least 4GB of space for the binary install. Once you've installed Ollama, you'll need additional space for storing the Large Language models, which can be tens to hundreds of GB in size. If your home directory doesn't have enough space, you can change where the binaries are installed, and where the models are stored.

### Changing Install Location

To install the Ollama application in a location different than your home directory, start the installer with the following flag

```powershell  theme={"system"}
OllamaSetup.exe /DIR="d:\some\location"
```

### Changing Model Location

To change where Ollama stores the downloaded models instead of using your home directory, set the environment variable `OLLAMA_MODELS` in your user account.

1. Start the Settings (Windows 11) or Control Panel (Windows 10) application and search for *environment variables*.

2. Click on *Edit environment variables for your account*.

3. Edit or create a new variable for your user account for `OLLAMA_MODELS` where you want the models stored

4. Click OK/Apply to save.

If Ollama is already running, Quit the tray application and relaunch it from the Start menu, or a new terminal started after you saved the environment variables.

## API Access

Here's a quick example showing API access from `powershell`

```powershell  theme={"system"}
(Invoke-WebRequest -method POST -Body '{"model":"llama3.2", "prompt":"Why is the sky blue?", "stream": false}' -uri http://localhost:11434/api/generate ).Content | ConvertFrom-json
```

## Troubleshooting

Ollama on Windows stores files in a few different locations. You can view them in
the explorer window by hitting `<Ctrl>+R` and type in:

* `explorer %LOCALAPPDATA%\Ollama` contains logs, and downloaded updates
  * *app.log* contains most resent logs from the GUI application
  * *server.log* contains the most recent server logs
  * *upgrade.log* contains log output for upgrades
* `explorer %LOCALAPPDATA%\Programs\Ollama` contains the binaries (The installer adds this to your user PATH)
* `explorer %HOMEPATH%\.ollama` contains models and configuration
* `explorer %TEMP%` contains temporary executable files in one or more `ollama*` directories

## Uninstall

The Ollama Windows installer registers an Uninstaller application. Under `Add or remove programs` in Windows Settings, you can uninstall Ollama.

<Note>
  If you have [changed the OLLAMA\_MODELS location](#changing-model-location), the installer will not remove your downloaded models
</Note>

## Standalone CLI

The easiest way to install Ollama on Windows is to use the `OllamaSetup.exe`
installer. It installs in your account without requiring Administrator rights.
We update Ollama regularly to support the latest models, and this installer will
help you keep up to date.

If you'd like to install or integrate Ollama as a service, a standalone
`ollama-windows-amd64.zip` zip file is available containing only the Ollama CLI
and GPU library dependencies for Nvidia. If you have an AMD GPU, also download
and extract the additional ROCm package `ollama-windows-amd64-rocm.zip` into the
same directory. This allows for embedding Ollama in existing applications, or
running it as a system service via `ollama serve` with tools such as
[NSSM](https://nssm.cc/).

<Note>
  If you are upgrading from a prior version, you should remove the old directories first.
</Note>

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Importing a Model

## Table of Contents

* [Importing a Safetensors adapter](#Importing-a-fine-tuned-adapter-from-Safetensors-weights)
* [Importing a Safetensors model](#Importing-a-model-from-Safetensors-weights)
* [Importing a GGUF file](#Importing-a-GGUF-based-model-or-adapter)
* [Sharing models on ollama.com](#Sharing-your-model-on-ollamacom)

## Importing a fine tuned adapter from Safetensors weights

First, create a `Modelfile` with a `FROM` command pointing at the base model you used for fine tuning, and an `ADAPTER` command which points to the directory with your Safetensors adapter:

```dockerfile  theme={"system"}
FROM <base model name>
ADAPTER /path/to/safetensors/adapter/directory
```

Make sure that you use the same base model in the `FROM` command as you used to create the adapter otherwise you will get erratic results. Most frameworks use different quantization methods, so it's best to use non-quantized (i.e. non-QLoRA) adapters. If your adapter is in the same directory as your `Modelfile`, use `ADAPTER .` to specify the adapter path.

Now run `ollama create` from the directory where the `Modelfile` was created:

```shell  theme={"system"}
ollama create my-model
```

Lastly, test the model:

```shell  theme={"system"}
ollama run my-model
```

Ollama supports importing adapters based on several different model architectures including:

* Llama (including Llama 2, Llama 3, Llama 3.1, and Llama 3.2);
* Mistral (including Mistral 1, Mistral 2, and Mixtral); and
* Gemma (including Gemma 1 and Gemma 2)

You can create the adapter using a fine tuning framework or tool which can output adapters in the Safetensors format, such as:

* Hugging Face [fine tuning framework](https://huggingface.co/docs/transformers/en/training)
* [Unsloth](https://github.com/unslothai/unsloth)
* [MLX](https://github.com/ml-explore/mlx)

## Importing a model from Safetensors weights

First, create a `Modelfile` with a `FROM` command which points to the directory containing your Safetensors weights:

```dockerfile  theme={"system"}
FROM /path/to/safetensors/directory
```

If you create the Modelfile in the same directory as the weights, you can use the command `FROM .`.

Now run the `ollama create` command from the directory where you created the `Modelfile`:

```shell  theme={"system"}
ollama create my-model
```

Lastly, test the model:

```shell  theme={"system"}
ollama run my-model
```

Ollama supports importing models for several different architectures including:

* Llama (including Llama 2, Llama 3, Llama 3.1, and Llama 3.2);
* Mistral (including Mistral 1, Mistral 2, and Mixtral);
* Gemma (including Gemma 1 and Gemma 2); and
* Phi3

This includes importing foundation models as well as any fine tuned models which have been *fused* with a foundation model.

## Importing a GGUF based model or adapter

If you have a GGUF based model or adapter it is possible to import it into Ollama. You can obtain a GGUF model or adapter by:

* converting a Safetensors model with the `convert_hf_to_gguf.py` from Llama.cpp;
* converting a Safetensors adapter with the `convert_lora_to_gguf.py` from Llama.cpp; or
* downloading a model or adapter from a place such as HuggingFace

To import a GGUF model, create a `Modelfile` containing:

```dockerfile  theme={"system"}
FROM /path/to/file.gguf
```

For a GGUF adapter, create the `Modelfile` with:

```dockerfile  theme={"system"}
FROM <model name>
ADAPTER /path/to/file.gguf
```

When importing a GGUF adapter, it's important to use the same base model as the base model that the adapter was created with. You can use:

* a model from Ollama
* a GGUF file
* a Safetensors based model

Once you have created your `Modelfile`, use the `ollama create` command to build the model.

```shell  theme={"system"}
ollama create my-model
```

## Quantizing a Model

Quantizing a model allows you to run models faster and with less memory consumption but at reduced accuracy. This allows you to run a model on more modest hardware.

Ollama can quantize FP16 and FP32 based models into different quantization levels using the `-q/--quantize` flag with the `ollama create` command.

First, create a Modelfile with the FP16 or FP32 based model you wish to quantize.

```dockerfile  theme={"system"}
FROM /path/to/my/gemma/f16/model
```

Use `ollama create` to then create the quantized model.

```shell  theme={"system"}
$ ollama create --quantize q4_K_M mymodel
transferring model data
quantizing F16 model to Q4_K_M
creating new layer sha256:735e246cc1abfd06e9cdcf95504d6789a6cd1ad7577108a70d9902fef503c1bd
creating new layer sha256:0853f0ad24e5865173bbf9ffcc7b0f5d56b66fd690ab1009867e45e7d2c4db0f
writing manifest
success
```

### Supported Quantizations

* `q8_0`

#### K-means Quantizations

* `q4_K_S`
* `q4_K_M`

## Sharing your model on ollama.com

You can share any model you have created by pushing it to [ollama.com](https://ollama.com) so that other users can try it out.

First, use your browser to go to the [Ollama Sign-Up](https://ollama.com/signup) page. If you already have an account, you can skip this step.

<img src="https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/signup.png?fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=d99f1340e6cfd85d36d49a444491cc63" alt="Sign-Up" width="40%" data-og-width="756" data-og-height="1192" data-path="images/signup.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/signup.png?w=280&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=8546e600b6c2b29ca91b23d837e9dc94 280w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/signup.png?w=560&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=786fba46e20fe8f9c2675abb620c7643 560w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/signup.png?w=840&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=06b9d9837022e7dbe9f9005f6f977eca 840w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/signup.png?w=1100&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=7ff9cc91091ffad52f8fb30a990d3089 1100w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/signup.png?w=1650&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=727fbd87da3a076b45794ea248f1afd3 1650w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/signup.png?w=2500&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=2ff3ce5e5197144725e860513cfe59e8 2500w" />

The `Username` field will be used as part of your model's name (e.g. `jmorganca/mymodel`), so make sure you are comfortable with the username that you have selected.

Now that you have created an account and are signed-in, go to the [Ollama Keys Settings](https://ollama.com/settings/keys) page.

Follow the directions on the page to determine where your Ollama Public Key is located.

<img src="https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/ollama-keys.png?fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=7ced4d97ecf6b115219f929a4914205e" alt="Ollama Keys" width="80%" data-og-width="1200" data-og-height="893" data-path="images/ollama-keys.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/ollama-keys.png?w=280&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=e3c7925a5a4f5dadafcf3908df55b97c 280w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/ollama-keys.png?w=560&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=66bb814e45ec58e6b15a4be890c6fec8 560w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/ollama-keys.png?w=840&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=44af18318e8469b719218a46305361d6 840w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/ollama-keys.png?w=1100&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=46d0cfa93938c243b8e41a169ceedb1f 1100w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/ollama-keys.png?w=1650&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=c235d8ed66c24171946e376eb5e6663e 1650w, https://mintcdn.com/ollama-9269c548/uieua2DvLKVQ74Ga/images/ollama-keys.png?w=2500&fit=max&auto=format&n=uieua2DvLKVQ74Ga&q=85&s=4844902d0289c2154b65d3d0339e9934 2500w" />

Click on the `Add Ollama Public Key` button, and copy and paste the contents of your Ollama Public Key into the text field.

To push a model to [ollama.com](https://ollama.com), first make sure that it is named correctly with your username. You may have to use the `ollama cp` command to copy
your model to give it the correct name. Once you're happy with your model's name, use the `ollama push` command to push it to [ollama.com](https://ollama.com).

```shell  theme={"system"}
ollama cp mymodel myuser/mymodel
ollama push myuser/mymodel
```

Once your model has been pushed, other users can pull and run it by using the command:

```shell  theme={"system"}
ollama run myuser/mymodel
```

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# null

## CPU only

```shell  theme={"system"}
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Nvidia GPU

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

### Install with Apt

1. Configure the repository

   ```shell  theme={"system"}
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
       | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
       | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
       | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   ```

2. Install the NVIDIA Container Toolkit packages

   ```shell  theme={"system"}
   sudo apt-get install -y nvidia-container-toolkit
   ```

### Install with Yum or Dnf

1. Configure the repository

   ```shell  theme={"system"}
   curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
       | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
   ```

2. Install the NVIDIA Container Toolkit packages

   ```shell  theme={"system"}
   sudo yum install -y nvidia-container-toolkit
   ```

### Configure Docker to use Nvidia driver

```shell  theme={"system"}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Start the container

```shell  theme={"system"}
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

<Note>
  If you're running on an NVIDIA JetPack system, Ollama can't automatically discover the correct JetPack version.
  Pass the environment variable `JETSON_JETPACK=5` or `JETSON_JETPACK=6` to the container to select version 5 or 6.
</Note>

## AMD GPU

To run Ollama using Docker with AMD GPUs, use the `rocm` tag and the following command:

```shell  theme={"system"}
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

## Vulkan Support

Vulkan is bundled into the `ollama/ollama` image.

```shell  theme={"system"}
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 -e OLLAMA_VULKAN=1 --name ollama ollama/ollama
```

## Run model locally

Now you can run a model:

```shell  theme={"system"}
docker exec -it ollama ollama run llama3.2
```

## Try different models

More models can be found on the [Ollama library](https://ollama.com/library).

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Troubleshooting

> How to troubleshoot issues encountered with Ollama

Sometimes Ollama may not perform as expected. One of the best ways to figure out what happened is to take a look at the logs. Find the logs on **Mac** by running the command:

```shell  theme={"system"}
cat ~/.ollama/logs/server.log
```

On **Linux** systems with systemd, the logs can be found with this command:

```shell  theme={"system"}
journalctl -u ollama --no-pager --follow --pager-end
```

When you run Ollama in a **container**, the logs go to stdout/stderr in the container:

```shell  theme={"system"}
docker logs <container-name>
```

(Use `docker ps` to find the container name)

If manually running `ollama serve` in a terminal, the logs will be on that terminal.

When you run Ollama on **Windows**, there are a few different locations. You can view them in the explorer window by hitting `<cmd>+R` and type in:

* `explorer %LOCALAPPDATA%\Ollama` to view logs. The most recent server logs will be in `server.log` and older logs will be in `server-#.log`
* `explorer %LOCALAPPDATA%\Programs\Ollama` to browse the binaries (The installer adds this to your user PATH)
* `explorer %HOMEPATH%\.ollama` to browse where models and configuration is stored
* `explorer %TEMP%` where temporary executable files are stored in one or more `ollama*` directories

To enable additional debug logging to help troubleshoot problems, first **Quit the running app from the tray menu** then in a powershell terminal

```powershell  theme={"system"}
$env:OLLAMA_DEBUG="1"
& "ollama app.exe"
```

Join the [Discord](https://discord.gg/ollama) for help interpreting the logs.

## LLM libraries

Ollama includes multiple LLM libraries compiled for different GPUs and CPU vector features. Ollama tries to pick the best one based on the capabilities of your system. If this autodetection has problems, or you run into other problems (e.g. crashes in your GPU) you can workaround this by forcing a specific LLM library. `cpu_avx2` will perform the best, followed by `cpu_avx` an the slowest but most compatible is `cpu`. Rosetta emulation under MacOS will work with the `cpu` library.

In the server log, you will see a message that looks something like this (varies from release to release):

```
Dynamic LLM libraries [rocm_v6 cpu cpu_avx cpu_avx2 cuda_v11 rocm_v5]
```

**Experimental LLM Library Override**

You can set OLLAMA\_LLM\_LIBRARY to any of the available LLM libraries to bypass autodetection, so for example, if you have a CUDA card, but want to force the CPU LLM library with AVX2 vector support, use:

```shell  theme={"system"}
OLLAMA_LLM_LIBRARY="cpu_avx2" ollama serve
```

You can see what features your CPU has with the following.

```shell  theme={"system"}
cat /proc/cpuinfo| grep flags | head -1
```

## Installing older or pre-release versions on Linux

If you run into problems on Linux and want to install an older version, or you'd like to try out a pre-release before it's officially released, you can tell the install script which version to install.

```shell  theme={"system"}
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.5.7 sh
```

## Linux tmp noexec

If your system is configured with the "noexec" flag where Ollama stores its temporary executable files, you can specify an alternate location by setting OLLAMA\_TMPDIR to a location writable by the user ollama runs as. For example OLLAMA\_TMPDIR=/usr/share/ollama/

## Linux docker

If Ollama initially works on the GPU in a docker container, but then switches to running on CPU after some period of time with errors in the server log reporting GPU discovery failures, this can be resolved by disabling systemd cgroup management in Docker. Edit `/etc/docker/daemon.json` on the host and add `"exec-opts": ["native.cgroupdriver=cgroupfs"]` to the docker configuration.

## NVIDIA GPU Discovery

When Ollama starts up, it takes inventory of the GPUs present in the system to determine compatibility and how much VRAM is available. Sometimes this discovery can fail to find your GPUs. In general, running the latest driver will yield the best results.

### Linux NVIDIA Troubleshooting

If you are using a container to run Ollama, make sure you've set up the container runtime first as described in [docker](./docker)

Sometimes the Ollama can have difficulties initializing the GPU. When you check the server logs, this can show up as various error codes, such as "3" (not initialized), "46" (device unavailable), "100" (no device), "999" (unknown), or others. The following troubleshooting techniques may help resolve the problem

* If you are using a container, is the container runtime working? Try `docker run --gpus all ubuntu nvidia-smi` - if this doesn't work, Ollama won't be able to see your NVIDIA GPU.
* Is the uvm driver loaded? `sudo nvidia-modprobe -u`
* Try reloading the nvidia\_uvm driver - `sudo rmmod nvidia_uvm` then `sudo modprobe nvidia_uvm`
* Try rebooting
* Make sure you're running the latest nvidia drivers

If none of those resolve the problem, gather additional information and file an issue:

* Set `CUDA_ERROR_LEVEL=50` and try again to get more diagnostic logs
* Check dmesg for any errors `sudo dmesg | grep -i nvrm` and `sudo dmesg | grep -i nvidia`

## AMD GPU Discovery

On linux, AMD GPU access typically requires `video` and/or `render` group membership to access the `/dev/kfd` device. If permissions are not set up correctly, Ollama will detect this and report an error in the server log.

When running in a container, in some Linux distributions and container runtimes, the ollama process may be unable to access the GPU. Use `ls -lnd /dev/kfd /dev/dri /dev/dri/*` on the host system to determine the **numeric** group IDs on your system, and pass additional `--group-add ...` arguments to the container so it can access the required devices. For example, in the following output `crw-rw---- 1 0  44 226,   0 Sep 16 16:55 /dev/dri/card0` the group ID column is `44`

If you are experiencing problems getting Ollama to correctly discover or use your GPU for inference, the following may help isolate the failure.

* `AMD_LOG_LEVEL=3` Enable info log levels in the AMD HIP/ROCm libraries. This can help show more detailed error codes that can help troubleshoot problems
* `OLLAMA_DEBUG=1` During GPU discovery additional information will be reported
* Check dmesg for any errors from amdgpu or kfd drivers `sudo dmesg | grep -i amdgpu` and `sudo dmesg | grep -i kfd`

## Multiple AMD GPUs

If you experience gibberish responses when models load across multiple AMD GPUs on Linux, see the following guide.

* [https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native\_linux/mgpu.html#mgpu-known-issues-and-limitations](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/mgpu.html#mgpu-known-issues-and-limitations)

## Windows Terminal Errors

Older versions of Windows 10 (e.g., 21H1) are known to have a bug where the standard terminal program does not display control characters correctly. This can result in a long string of strings like `←[?25h←[?25l` being displayed, sometimes erroring with `The parameter is incorrect` To resolve this problem, please update to Win 10 22H1 or newer.
