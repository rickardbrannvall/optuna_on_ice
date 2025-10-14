## Running on the Cluster with JupyterLab

This guide explains how to use this repository on a cluster environment using a JupyterLab pod.

### 1. Launch a JupyterLab Pod

Start by creating a JupyterLab pod using the ICE standard template charts for the jupyternode apps. You can use the default settings, but check

- Select the latest TensorFlow image from Docker Hub (e.g., `tensorflow/tensorflow:latest-gpu`)
- Choose a unique subdomain for icedc.se, e.g., `[myjupyterpod]-[mynamespace].icedc.se`
- Attach one or more GPU if available
- Optionally mount a shared persistent storage volume if you want to store results or work across sessions

Shared persistent storage must be set-up separately before you create your JupyterLab pod.

### 2. Access JupyterLab

Once the pod is running, open JupyterLab in your browser through the choosen subdomain on icedc.se. 

Navigate to your preferred working directory — ideally one located on the shared persistent volume if mounted.

### 3. Clone the Repository

Open a terminal in JupyterLab and run the following commands:

$ git clone https://github.com/rickardbrannvall/optuna_on_ice.git

$ cd optuna_on_ice

### 4. Install Dependencies

Run the installation script to install all required libraries:

$ bash install.sh

### 5. Check GPU Availability
To verify how many GPUs are available to your pod, run:

$ nvidia-smi

### 6. Run the Training Script
Start the training process by running one of the following commands.

#### On CPU without augmentation: 

$ python cifar10_resnet18_runner.py --n_trials 2 --study_name "cifar10_resnet" --augment False

#### On GPU and with augmentation

$ python cifar10_resnet18_runner.py --gpu 1 --n_trials 2 --study_name "cifar10_resnet_augment" --augment True

#### In parallel on 8 GPUs

$ parallel --ungroup python cifar10_resnet18_runner.py --gpu {} --n_trials 2 --study_name "cifar10_resnet" ::: {0..7}

### 7. Monitor Training Progress
Once training has started, you can monitor progress by opening the provided Jupyter notebook in the repository. It will visualize training metrics and optimization history.

- Notebook: study_visualization.ipynb
