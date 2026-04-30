#!/bin/bash

# Define directories
SDK_DIR="/data/sdk"
DOWNLOAD_DIR="${HOME}/Documents"
VIDEO_DIR="/data/video"
ZIP_FILE="v2.26.0.240828.zip"

# Create the necessary directories if they do not exist
sudo mkdir -p "$SDK_DIR"
sudo mkdir -p "$VIDEO_DIR"

# Download the zip file
echo "Downloading SDK zip file..."
sudo curl -L -o "$SDK_DIR/$ZIP_FILE" "https://huggingface.co/datasets/kaiwei0323/my-sdk/resolve/main/v2.26.0.240828.zip"

# Check if the zip file exists before attempting to unzip
if [ -f "$SDK_DIR/$ZIP_FILE" ]; then
  echo "Extracting zip file..."
  sudo unzip "$SDK_DIR/$ZIP_FILE" -d "$SDK_DIR"
  echo "SDK extracted successfully."
  # Delete the zip file after extraction
  sudo rm "$SDK_DIR/$ZIP_FILE"
  echo "ZIP file deleted."
else
  echo "Error: ZIP file not found at $SDK_DIR/$ZIP_FILE. Skipping extraction."
fi

# Download the video files into the correct directory
echo "Downloading video files..."
sudo curl -L -o "$VIDEO_DIR/brain_tumor.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/brain_tumor.mp4"
sudo curl -L -o "$VIDEO_DIR/fall.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/fall.mp4"
sudo curl -L -o "$VIDEO_DIR/freeway.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/freeway.mp4"
sudo curl -L -o "$VIDEO_DIR/med_ppe.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/med_ppe.mp4"
sudo curl -L -o "$VIDEO_DIR/ppe.mp4" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/ppe.mp4"

echo "Video files downloaded successfully to $VIDEO_DIR"

# Set up DSP environment variables and add to .bashrc
echo "Setting up DSP environment variables..."

# Define SNPE paths
SNPE_ROOT="$SDK_DIR/v2.26.0.240828/qairt/2.26.0.240828"
TUTORIALS_DIR="$DOWNLOAD_DIR/SNPE_Flask/Tutorials"

# Create the DSP environment configuration
cat >> ~/.bashrc << 'EOF'

# SNPE DSP Environment Variables
export SNPE_ROOT="/data/sdk/v2.26.0.240828/qairt/2.26.0.240828"
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4"
# Order matters: SNPE host libs first, then DSP skels, then system libs
export LD_LIBRARY_PATH="$SNPE_LIBRARY_PATH:$ADSP_LIBRARY_PATH:/usr/lib/qcm6490:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}"

EOF

# Source the updated .bashrc for current session
source ~/.bashrc

echo "DSP environment variables added to .bashrc"
echo "SNPE_ROOT: $SNPE_ROOT"
echo "ADSP_LIBRARY_PATH: $ADSP_LIBRARY_PATH"

# Add Qualcomm IoT PPA (for QCOM GStreamer plugins, etc.)
if [ ! -f /etc/apt/sources.list.d/ubuntu-qcom-iot-ubuntu-qcom-ppa-noble.list ]; then
  sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
fi

sudo apt update

# Install necessary packages
sudo apt install -y \
    gstreamer1.0-plugins-qcom-base \
    gstreamer1.0-plugins-qcom-good \
    gstreamer1.0-plugins-qcom-bad \
    gstreamer1.0-plugins-qcom \
    gstreamer1.0-plugins-qcom-vtransform \
    gstreamer1.0-qcom-sample-apps \
    python3-pip \
    python3-venv \
    python3-pybind11 \
    cmake \
    python3-flask \
    python3-opencv \
    python3-paho-mqtt \
    gir1.2-gst-plugins-base-1.0 \
    mosquitto \
    mosquitto-clients

# Start and enable mosquitto service
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# PyTorch (requested). Note: uses --break-system-packages on Ubuntu 24.04+ (PEP 668).
python3 -m pip install --break-system-packages torch torchvision

echo "Setup complete!"
