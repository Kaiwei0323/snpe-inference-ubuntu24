#!/bin/bash
#
# Setup script for SNPE_Flask Tutorials.
#
# Failures are no longer silently ignored. Standard Ubuntu packages and
# Qualcomm-PPA packages are installed in separate apt transactions so that
# a temporary PPA outage (e.g. HTTP 503) doesn't block pip/flask/mosquitto
# from being installed.

set -uo pipefail

# Define directories
SDK_DIR="/data/sdk"
DOWNLOAD_DIR="${HOME}/Documents"
VIDEO_DIR="/data/video"
ZIP_FILE="v2.26.0.240828.zip"

# Track non-fatal failures so we can summarize at the end and exit non-zero.
FAILURES=()
record_failure() {
    FAILURES+=("$1")
    echo "  [WARN] $1" >&2
}

# Create the necessary directories if they do not exist
sudo mkdir -p "$SDK_DIR"
sudo mkdir -p "$VIDEO_DIR"

# Download the SDK zip (skip if already extracted)
SNPE_ROOT="$SDK_DIR/v2.26.0.240828/qairt/2.26.0.240828"
if [ -d "$SNPE_ROOT" ]; then
    echo "SDK already extracted at $SNPE_ROOT, skipping download."
else
    echo "Downloading SDK zip file..."
    if sudo curl -fL -o "$SDK_DIR/$ZIP_FILE" \
        "https://huggingface.co/datasets/kaiwei0323/my-sdk/resolve/main/v2.26.0.240828.zip"; then
        echo "Extracting zip file..."
        if sudo unzip -q "$SDK_DIR/$ZIP_FILE" -d "$SDK_DIR"; then
            echo "SDK extracted successfully."
            sudo rm -f "$SDK_DIR/$ZIP_FILE"
            echo "ZIP file deleted."
        else
            record_failure "Failed to extract $SDK_DIR/$ZIP_FILE"
        fi
    else
        record_failure "Failed to download SDK zip file"
    fi
fi

# Download the video files into the correct directory (idempotent: -C - resumes).
echo "Downloading video files..."
VIDEO_URLS=(
    "brain_tumor.mp4"
    "fall.mp4"
    "freeway.mp4"
    "med_ppe.mp4"
    "ppe.mp4"
)
for v in "${VIDEO_URLS[@]}"; do
    dest="$VIDEO_DIR/$v"
    if [ -s "$dest" ]; then
        echo "  $v already present, skipping."
        continue
    fi
    if ! sudo curl -fL -o "$dest" \
        "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/$v"; then
        record_failure "Failed to download video $v"
    fi
done
echo "Video files step finished (target: $VIDEO_DIR)."

# Set up DSP environment variables in .bashrc (idempotent).
echo "Setting up DSP environment variables..."
TUTORIALS_DIR="$DOWNLOAD_DIR/SNPE_Flask/Tutorials"
BASHRC_MARK="# >>> SNPE DSP Environment Variables >>>"

if grep -qF "$BASHRC_MARK" "$HOME/.bashrc" 2>/dev/null; then
    echo "DSP env vars already present in ~/.bashrc, skipping append."
else
    cat >> "$HOME/.bashrc" << 'EOF'

# >>> SNPE DSP Environment Variables >>>
export SNPE_ROOT="/data/sdk/v2.26.0.240828/qairt/2.26.0.240828"
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4"
# Order matters: SNPE host libs first, then DSP skels, then system libs
export LD_LIBRARY_PATH="$SNPE_LIBRARY_PATH:$ADSP_LIBRARY_PATH:/usr/lib/qcm6490:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
# <<< SNPE DSP Environment Variables <<<
EOF
    echo "DSP environment variables added to .bashrc"
fi

echo "SNPE_ROOT: $SNPE_ROOT"
echo "ADSP_LIBRARY_PATH: $SNPE_ROOT/lib/hexagon-v68/unsigned"

# Ensure ~/.local/bin is on PATH so pip-user-installed entry points
# (torchrun, isympy, proton, etc.) are runnable from any shell.
LOCAL_BIN_MARK="# >>> SNPE_Flask local bin PATH >>>"
if grep -qF "$LOCAL_BIN_MARK" "$HOME/.bashrc" 2>/dev/null; then
    echo "~/.local/bin already added to PATH in ~/.bashrc, skipping."
else
    cat >> "$HOME/.bashrc" << 'EOF'

# >>> SNPE_Flask local bin PATH >>>
# Ensure pip --user installed scripts (e.g. torchrun) are on PATH.
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
# <<< SNPE_Flask local bin PATH <<<
EOF
    echo "Added ~/.local/bin to PATH in ~/.bashrc"
fi
# Also make it available for the rest of this script run.
case ":${PATH:-}:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:${PATH:-}" ;;
esac

# Add Qualcomm IoT PPA (for QCOM GStreamer plugins, etc.)
if [ ! -f /etc/apt/sources.list.d/ubuntu-qcom-iot-ubuntu-qcom-ppa-noble.list ]; then
    if ! sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa; then
        record_failure "Failed to add Qualcomm IoT PPA"
    fi
fi

echo "Running apt update..."
if ! sudo apt update; then
    record_failure "apt update reported errors (continuing)"
fi

# --- Group 1: Standard Ubuntu packages ---
# These come from the main Ubuntu archive and should always be available,
# even if the Qualcomm PPA is down. Install them FIRST so pip/flask/mosquitto
# work regardless of PPA status.
echo "Installing standard Ubuntu packages..."
STANDARD_PKGS=(
    snpe-tools
    python3-pip
    python3-venv
    python3-pybind11
    cmake
    python3-flask
    python3-opencv
    python3-paho-mqtt
    mosquitto
    mosquitto-clients
    # QRTR / cDSP plumbing required for SNPE DSP runtime on QCS6490:
    #   tqftpserv : serves files from rootfs to the cDSP over QRTR
    #   rmtfs     : remote filesystem service some skels need
    tqftpserv
    rmtfs
)
if ! sudo apt install -y "${STANDARD_PKGS[@]}"; then
    record_failure "Failed to install one or more standard Ubuntu packages"
fi

# GstApp's introspection typelib is needed by pipelines/BasePipeline.py.
# Install it from Ubuntu's archive so a Qualcomm PPA outage doesn't block app startup.
echo "Installing GStreamer introspection packages..."
GST_INTROSPECTION_PKGS=(
    gir1.2-gst-plugins-base-1.0
)
if ! sudo apt install -y -t noble-updates "${GST_INTROSPECTION_PKGS[@]}"; then
    record_failure "Failed to install GStreamer introspection packages"
fi

# --- Group 2: Qualcomm PPA packages ---
# These can fail if the Qualcomm PPA is temporarily unavailable (e.g. HTTP 503).
# A failure here is non-fatal: re-run this script later when the PPA is back.
echo "Installing Qualcomm PPA packages..."
QCOM_PKGS=(
    gstreamer1.0-plugins-qcom-base
    gstreamer1.0-plugins-qcom-good
    gstreamer1.0-plugins-qcom-bad
    gstreamer1.0-plugins-qcom
    gstreamer1.0-plugins-qcom-vtransform
    gstreamer1.0-qcom-sample-apps
)
if ! sudo apt install -y "${QCOM_PKGS[@]}"; then
    record_failure "Failed to install Qualcomm PPA packages (PPA may be down; re-run later)"
fi

# Start and enable mosquitto only if it was actually installed.
if command -v mosquitto >/dev/null 2>&1 || dpkg -s mosquitto >/dev/null 2>&1; then
    sudo systemctl start mosquitto || record_failure "Failed to start mosquitto"
    sudo systemctl enable mosquitto || record_failure "Failed to enable mosquitto"
else
    record_failure "mosquitto not installed; skipping systemctl start/enable"
fi

# --- SNPE DSP runtime wiring (QCS6490 / cDSP) -------------------------------
# Several pieces are needed for SNPE to actually reach the Hexagon cDSP instead
# of silently falling back to CPU. We set them up idempotently.
echo "Configuring SNPE DSP runtime..."

# (a) The Qualcomm packages ship /usr/lib/aarch64-linux-gnu/libcdsprpc.so (cDSP
#     FastRPC), but the SDK's host stubs dlopen the legacy name libadsprpc.so.
#     They're ABI-compatible, so symlink one to the other.
if [ -e /usr/lib/aarch64-linux-gnu/libcdsprpc.so ] && [ ! -e /usr/lib/libadsprpc.so ]; then
    sudo ln -sfn /usr/lib/aarch64-linux-gnu/libcdsprpc.so /usr/lib/libadsprpc.so \
        || record_failure "Failed to create /usr/lib/libadsprpc.so symlink"
fi

# (b) /dev/fastrpc-cdsp is mode crw-rw-r-- root:fastrpc. To submit work to the
#     cDSP we need rw, so the current user must be in the 'fastrpc' group.
#     The change only takes effect in shells started AFTER setup.sh.
TARGET_USER="${SUDO_USER:-$USER}"
if getent group fastrpc >/dev/null 2>&1; then
    if id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx fastrpc; then
        echo "  $TARGET_USER already in 'fastrpc' group."
    else
        sudo usermod -aG fastrpc "$TARGET_USER" \
            && echo "  Added $TARGET_USER to 'fastrpc' group (re-login required)." \
            || record_failure "Failed to add $TARGET_USER to 'fastrpc' group"
    fi
else
    record_failure "'fastrpc' group not present (is qcom-fastrpc1 installed?)"
fi

# (c) Bring up the QRTR/DSP file-service daemons in the right order. These are
#     what the cDSP firmware reaches back to for skel files and PD info.
for svc in qrtr-ns pd-mapper tqftpserv rmtfs; do
    if systemctl list-unit-files "$svc.service" >/dev/null 2>&1 \
            && systemctl cat "$svc.service" >/dev/null 2>&1; then
        sudo systemctl enable --now "$svc" >/dev/null 2>&1 \
            || record_failure "Failed to enable/start $svc"
    fi
done

# (d) At boot the kernel auto-loads the cDSP firmware BEFORE these daemons are
#     up, so the firmware never learns who to talk to. Install a one-shot unit
#     that re-bounces the cDSP after the daemons are ready, so SNPE DSP works
#     across reboots without manual intervention.
CDSP_UNIT=/etc/systemd/system/cdsp-rehandshake.service
if [ ! -f "$CDSP_UNIT" ]; then
    sudo tee "$CDSP_UNIT" >/dev/null <<'UNIT'
[Unit]
Description=Re-attach cDSP firmware after QRTR/FastRPC services are up
# The cDSP is auto-booted by the kernel before any of these daemons exist, so
# its first attach never sees tqftpserv/pd-mapper. Bouncing it here forces a
# clean handshake with the now-running services.
After=qrtr-ns.service pd-mapper.service tqftpserv.service
Wants=qrtr-ns.service pd-mapper.service tqftpserv.service
ConditionPathExists=/sys/class/remoteproc/remoteproc0/state

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c '\
    state_file=/sys/class/remoteproc/remoteproc0/state; \
    fw=$(cat /sys/class/remoteproc/remoteproc0/firmware 2>/dev/null); \
    case "$fw" in *cdsp*) ;; *) echo "remoteproc0 is not cDSP ($fw), skipping"; exit 0;; esac; \
    echo stop > "$state_file"; sleep 2; \
    echo start > "$state_file"; sleep 2; \
    echo "cDSP state: $(cat $state_file)"'

[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload \
        && sudo systemctl enable cdsp-rehandshake.service >/dev/null 2>&1 \
        || record_failure "Failed to install cdsp-rehandshake.service"
    echo "  Installed cdsp-rehandshake.service (runs on every boot)."

    # Trigger it now so the current session benefits without a reboot.
    sudo systemctl start cdsp-rehandshake.service \
        || record_failure "Failed to run cdsp-rehandshake now"
fi

# PyTorch (requested). Only attempt if pip is available.
# Uses --break-system-packages on Ubuntu 24.04+ (PEP 668).
if python3 -m pip --version >/dev/null 2>&1; then
    if ! python3 -m pip install --break-system-packages torch torchvision; then
        record_failure "pip install torch torchvision failed"
    fi
else
    record_failure "pip not available; skipping torch/torchvision install"
fi

# --- Summary ---
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "Setup complete!"
    exit 0
else
    echo ""
    echo "Setup finished with ${#FAILURES[@]} warning(s):"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "You can safely re-run this script after fixing the underlying issue"
    echo "(e.g. wait for the Qualcomm PPA to come back online)."
    exit 1
fi
