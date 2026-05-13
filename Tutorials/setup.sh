#!/bin/bash
# Setup script for SNPE_Flask Tutorials on QCS6490 / Ubuntu 24.04.
# Idempotent: safe to re-run. Failures are non-fatal but summarised at the end.

set -uo pipefail

# =============================================================================
#                                  Config
# =============================================================================
SDK_VERSION="v2.26.0.240828"
SDK_DIR="/data/sdk"
SNPE_ROOT="$SDK_DIR/$SDK_VERSION/qairt/2.26.0.240828"
SDK_ZIP_URL="https://huggingface.co/datasets/kaiwei0323/my-sdk/resolve/main/$SDK_VERSION.zip"

VIDEO_DIR="/data/video"
VIDEO_BASE_URL="https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main"
VIDEO_FILES=(brain_tumor.mp4 fall.mp4 freeway.mp4 med_ppe.mp4 ppe.mp4)

TARGET_USER="${SUDO_USER:-$USER}"

# =============================================================================
#                                 Helpers
# =============================================================================
FAILURES=()
warn()    { printf '  [WARN] %s\n' "$*" >&2; FAILURES+=("$*"); }
section() { printf '\n=== %s ===\n' "$*"; }

# Idempotent curl: skips if dest already exists with non-zero size.
fetch() {
    [ -s "$2" ] && return 0
    sudo curl -fL -o "$2" "$1"
}

# `apt install -y` with failure tracking.
apt_install() {
    local label="$1"; shift
    section "Installing $label"
    sudo apt install -y "$@" || warn "Failed to install $label"
}

# Enable+start a systemd unit, only if it exists on disk.
enable_service() {
    systemctl list-unit-files "$1.service" >/dev/null 2>&1 || return 0
    sudo systemctl enable --now "$1" >/dev/null 2>&1 || warn "Failed to enable/start $1"
}

# Replace (or insert) a marker-bracketed block in ~/.bashrc. Block content
# is read from stdin. Markers are "# >>> NAME >>>" ... "# <<< NAME <<<".
upsert_bashrc_block() {
    local name="$1"
    local content
    content=$(cat)
    BLOCK_NAME="$name" BLOCK_CONTENT="$content" python3 - <<'PY'
import os, re, pathlib
name    = os.environ["BLOCK_NAME"]
content = os.environ["BLOCK_CONTENT"]
p = pathlib.Path.home() / ".bashrc"
text = p.read_text() if p.exists() else ""
beg, end = re.escape(f"# >>> {name} >>>"), re.escape(f"# <<< {name} <<<")
text = re.sub(rf"\n*{beg}.*?{end}\n?", "\n", text, flags=re.DOTALL)
text = re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"
text += f"\n# >>> {name} >>>\n{content}\n# <<< {name} <<<\n"
p.write_text(text)
PY
}

# Strip any legacy unmarked SNPE block left by old versions of this script.
strip_legacy_snpe_bashrc() {
    python3 - <<'PY'
import re, pathlib
p = pathlib.Path.home() / ".bashrc"
if not p.exists(): raise SystemExit
text = p.read_text()
text = re.sub(
    r"\n*# SNPE DSP Environment Variables\n"
    r"export SNPE_ROOT=.*?\nexport ADSP_LIBRARY_PATH=.*?\n"
    r"export SNPE_LIBRARY_PATH=.*?\n# Order matters:[^\n]*\n"
    r"export LD_LIBRARY_PATH=.*?\n",
    "\n", text, flags=re.DOTALL,
)
p.write_text(re.sub(r"\n{3,}", "\n\n", text))
PY
}

# =============================================================================
#                              1. SNPE SDK
# =============================================================================
section "SNPE SDK"
sudo mkdir -p "$SDK_DIR" "$VIDEO_DIR"
if [ -d "$SNPE_ROOT" ]; then
    echo "SDK already extracted at $SNPE_ROOT"
else
    zip="$SDK_DIR/$SDK_VERSION.zip"
    echo "Downloading $SDK_ZIP_URL"
    if fetch "$SDK_ZIP_URL" "$zip" && sudo unzip -q "$zip" -d "$SDK_DIR"; then
        sudo rm -f "$zip"
        echo "Extracted to $SNPE_ROOT"
    else
        warn "SDK download or extract failed"
    fi
fi

# =============================================================================
#                            2. Demo videos
# =============================================================================
section "Demo videos"
for v in "${VIDEO_FILES[@]}"; do
    if [ -s "$VIDEO_DIR/$v" ]; then
        echo "  $v already present"
    else
        fetch "$VIDEO_BASE_URL/$v" "$VIDEO_DIR/$v" || warn "Failed to download $v"
    fi
done

# =============================================================================
#                       3. Shell environment (~/.bashrc)
# =============================================================================
# We deliberately do NOT touch LD_LIBRARY_PATH. The cDSP firmware loads its
# skels from /usr/lib/rfsa/adsp/ (libsnpe1 == SNPE 2.43). If we prepend the
# SDK 2.26 lib/ dirs, libsnpehelper.so loads the 2.26 host libSNPE.so and the
# 2.43 DSP skel rejects the handshake -> silent CPU fallback.
# We keep SNPE_ROOT pointing at the SDK for HOST-side tooling only
# (snpe-onnx-to-dlc, snpe-dlc-quantize, snpe-platform-validator, etc.).
section "Shell environment"
strip_legacy_snpe_bashrc

upsert_bashrc_block "SNPE DSP Environment Variables" <<EOF
export SNPE_ROOT="$SNPE_ROOT"
case ":\$PATH:" in
    *":\$SNPE_ROOT/bin/aarch64-ubuntu-gcc9.4:"*) ;;
    *) export PATH="\$SNPE_ROOT/bin/aarch64-ubuntu-gcc9.4:\$PATH" ;;
esac
EOF

upsert_bashrc_block "SNPE_Flask local bin PATH" <<'EOF'
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
EOF

# Apply both PATH additions to the rest of this script run too.
case ":${PATH:-}:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:${PATH:-}" ;; esac
case ":${PATH:-}:" in *":$SNPE_ROOT/bin/aarch64-ubuntu-gcc9.4:"*) ;; *) export PATH="$SNPE_ROOT/bin/aarch64-ubuntu-gcc9.4:${PATH:-}" ;; esac

echo "SNPE_ROOT=$SNPE_ROOT"

# =============================================================================
#                          4. APT repositories
# =============================================================================
section "APT repositories"
PPA_LIST=/etc/apt/sources.list.d/ubuntu-qcom-iot-ubuntu-qcom-ppa-noble.list
if [ ! -f "$PPA_LIST" ]; then
    sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa || warn "Failed to add Qualcomm IoT PPA"
fi
sudo apt update || warn "apt update reported errors"

# =============================================================================
#                  5. APT packages (split, PPA outage tolerant)
# =============================================================================
# Standard Ubuntu archive packages first - these are always available and
# include pip/flask/mosquitto plus the QRTR/cDSP plumbing for SNPE DSP.
apt_install "standard Ubuntu packages" \
    snpe-tools \
    python3-pip python3-venv python3-pybind11 cmake \
    python3-flask python3-opencv python3-paho-mqtt \
    mosquitto mosquitto-clients \
    tqftpserv rmtfs

# GstApp introspection typelib - pulled from Ubuntu main (NOT the qcom PPA),
# so it survives PPA outages. Required by pipelines/BasePipeline.py.
sudo apt install -y -t noble-updates gir1.2-gst-plugins-base-1.0 \
    || warn "Failed to install gir1.2-gst-plugins-base-1.0"

# Qualcomm PPA packages. PPA can 503 - non-fatal if so; just re-run later.
apt_install "Qualcomm GStreamer plugins (PPA)" \
    gstreamer1.0-plugins-qcom-base \
    gstreamer1.0-plugins-qcom-good \
    gstreamer1.0-plugins-qcom-bad \
    gstreamer1.0-plugins-qcom \
    gstreamer1.0-plugins-qcom-vtransform \
    gstreamer1.0-qcom-sample-apps

# =============================================================================
#                          6. Mosquitto service
# =============================================================================
section "Mosquitto"
if dpkg -s mosquitto >/dev/null 2>&1; then
    enable_service mosquitto
else
    warn "mosquitto not installed; skipping"
fi

# =============================================================================
#               7. SNPE DSP runtime wiring (QCS6490 / cDSP)
# =============================================================================
# Each step below is independently needed for SNPE to reach the Hexagon cDSP.
section "SNPE DSP runtime wiring"

# 7a) libadsprpc.so symlink. SDK host stubs dlopen the legacy aDSP name,
#     QCS6490 Ubuntu only ships cDSP. They're ABI-compatible.
if [ -e /usr/lib/aarch64-linux-gnu/libcdsprpc.so ] && [ ! -e /usr/lib/libadsprpc.so ]; then
    sudo ln -sfn /usr/lib/aarch64-linux-gnu/libcdsprpc.so /usr/lib/libadsprpc.so \
        || warn "Failed to create /usr/lib/libadsprpc.so symlink"
fi

# 7b) User must be in 'fastrpc' group to open /dev/fastrpc-cdsp rw.
#     Group change requires a fresh login shell to take effect.
if getent group fastrpc >/dev/null 2>&1; then
    if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx fastrpc; then
        echo "  $TARGET_USER already in 'fastrpc' group"
    elif sudo usermod -aG fastrpc "$TARGET_USER"; then
        echo "  Added $TARGET_USER to 'fastrpc' group (re-login required)"
    else
        warn "Failed to add $TARGET_USER to 'fastrpc' group"
    fi
else
    warn "'fastrpc' group missing (is qcom-fastrpc1 installed?)"
fi

# 7c) QRTR / file-service daemons the cDSP firmware reaches back to.
for svc in qrtr-ns pd-mapper tqftpserv rmtfs; do
    enable_service "$svc"
done

# 7d) At boot the kernel auto-loads the cDSP firmware BEFORE the daemons in
#     7c are running, so the firmware never learns who serves its file
#     requests. Install a one-shot unit that bounces the cDSP after the
#     daemons are up. We find the cDSP by firmware NAME, not by remoteproc
#     index, because remoteproc numbering isn't stable across reboots.
section "cdsp-rehandshake.service"
sudo tee /etc/systemd/system/cdsp-rehandshake.service >/dev/null <<'UNIT'
[Unit]
Description=Re-attach cDSP firmware after QRTR/FastRPC services are up
After=qrtr-ns.service pd-mapper.service tqftpserv.service rmtfs.service
Wants=qrtr-ns.service pd-mapper.service tqftpserv.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c '\
    for rp in /sys/class/remoteproc/remoteproc*; do \
        fw=$(cat "$rp/firmware" 2>/dev/null) || continue; \
        case "$fw" in *cdsp*) \
            echo "Bouncing cDSP at $rp (firmware=$fw)"; \
            echo stop  > "$rp/state"; sleep 2; \
            echo start > "$rp/state"; sleep 2; \
            echo "  state now: $(cat $rp/state)"; \
            ;; \
        esac; \
    done'

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable cdsp-rehandshake.service >/dev/null 2>&1 \
    || warn "Failed to enable cdsp-rehandshake.service"
sudo systemctl restart cdsp-rehandshake.service \
    || warn "Failed to run cdsp-rehandshake now"

# =============================================================================
#                            8. PyTorch (pip)
# =============================================================================
section "PyTorch"
if python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages torch torchvision \
        || warn "pip install torch torchvision failed"
else
    warn "pip not available; skipping torch/torchvision install"
fi

# =============================================================================
#                              Summary
# =============================================================================
echo ""
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "Setup complete."
    exit 0
fi
echo "Setup finished with ${#FAILURES[@]} warning(s):"
printf '  - %s\n' "${FAILURES[@]}"
echo "You can safely re-run this script after the underlying issue clears."
exit 1
