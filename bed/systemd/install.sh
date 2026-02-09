#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BED_BIN="${SCRIPT_DIR}/../bin/bed"

install_system() {
    echo "Installing system-wide bed service..."

    # Build if needed
    if [ ! -f "$BED_BIN" ]; then
        echo "Building bed..."
        cd "$SCRIPT_DIR/.."
        go build -o bin/bed ./bin/
    fi

    # Install binary
    sudo cp "$BED_BIN" /usr/local/bin/bed
    sudo chmod 755 /usr/local/bin/bed

    # Install service
    sudo cp "$SCRIPT_DIR/bed.service" /etc/systemd/system/bed@.service
    sudo systemctl daemon-reload

    echo "Installed. Enable with: sudo systemctl enable --now bed@\$USER"
}

install_user() {
    echo "Installing user bed service..."

    # Create user bin dir
    mkdir -p ~/.local/bin
    mkdir -p ~/.config/systemd/user

    # Build if needed
    if [ ! -f "$BED_BIN" ]; then
        echo "Building bed..."
        cd "$SCRIPT_DIR/.."
        go build -o bin/bed ./bin/
    fi

    # Install binary
    cp "$BED_BIN" ~/.local/bin/bed
    chmod 755 ~/.local/bin/bed

    # Install service
    cp "$SCRIPT_DIR/bed-user.service" ~/.config/systemd/user/bed.service
    systemctl --user daemon-reload

    echo "Installed. Enable with: systemctl --user enable --now bed"
}

case "${1:-user}" in
    system)
        install_system
        ;;
    user)
        install_user
        ;;
    *)
        echo "Usage: $0 [user|system]"
        exit 1
        ;;
esac
