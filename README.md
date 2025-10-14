## Running on the Cluster with JupyterLab

This guide explains how to use this repository on a cluster environment using a JupyterLab pod.

### 1. Launch a JupyterLab Pod

Start by creating a JupyterLab pod using your cluster’s standard template for apps. You can use the default settings, but make sure to:

- Select the latest TensorFlow image from Docker Hub (e.g., `tensorflow/tensorflow:latest-jupyter`)
- Optionally mount a persistent volume if you want to store results or work across sessions

### 2. Access JupyterLab

Once the pod is running, open JupyterLab in your browser. Navigate to your preferred working directory—ideally one located on the persistent volume if mounted.

### 3. Clone the Repository

Open a terminal in JupyterLab and run the following commands:

git clone https://github.com/rickardbrannvall/optuna_on_ice.git
cd optuna_on_ice

### 4. Install Dependencies

Run the installation script to install all required libraries:
bash install.sh

### 5. Check GPU Availability
To verify how many GPUs are available to your pod, run:

nvidia-smi

### 6. Run the Training Script
Start the training process by executing:

python train.py

### 7. Monitor Training Progress
Once training has started, you can monitor progress by opening the provided Jupyter notebook in the repository. It will visualize training metrics and optimization history.

- Notebook: study_visualization.ipynb


