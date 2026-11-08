#!/usr/bin/env bash
# Start the translator automatically at power-on, for the glasses.
# Run from inside the repo, after setup_pi.sh:
#
#     bash install_service.sh                        # install and start
#     bash install_service.sh --button-pin 17        # anything after is passed to main.py
#     bash install_service.sh --model landmark_model_firstperson.joblib
#     bash install_service.sh --uninstall            # stop it and remove it
#
# It runs `python main.py --headless` as a systemd user service, restarted if
# it crashes, and kept running without anyone logged in (loginctl linger).
# ASL_* variables set when you install (ASL_CAMERA, ASL_CAMERA_EV, ASL_TTS,
# ASL_PIPER_MODEL, ASL_DEVICE) are carried into the service.
#
# While it runs it holds the camera, so stop it before evaluate.py,
# collect_data.py, bench_pi.py or a windowed main.py:
#
#     systemctl --user stop asl-glasses      # until the next boot
#     systemctl --user start asl-glasses
#     journalctl --user -u asl-glasses -f    # what it is printing
set -euo pipefail

NAME=asl-glasses
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/$NAME.service"

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f main.py && -f requirements.txt ]] || die "run this from inside the ASL-Glasses folder"
REPO="$(pwd)"

if [[ "${1:-}" == "--uninstall" ]]; then
    say "Removing the $NAME service"
    systemctl --user disable --now "$NAME" 2>/dev/null || true
    rm -f "$UNIT"
    systemctl --user daemon-reload
    echo "Removed. (Linger is left on; 'loginctl disable-linger' turns it off.)"
    exit 0
fi

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || die "no .venv here -- run 'bash setup_pi.sh' first"

# systemd's own quoting: double quotes, with backslash and quote escaped, and
# % doubled (systemd reads %h, %u... as specifiers). Not the shell's quoting.
sdquote() {
    local s="${1//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//%/%%}"
    printf '"%s"' "$s"
}

# Every argument after the script name goes to main.py.
EXTRA=""
for arg in "$@"; do
    EXTRA+=" $(sdquote "$arg")"
done

ENV_LINES=""
for var in ASL_CAMERA ASL_CAMERA_EV ASL_TTS ASL_PIPER_MODEL ASL_DEVICE; do
    if [[ -n "${!var:-}" ]]; then
        ENV_LINES+="Environment=$(sdquote "$var=${!var}")"$'\n'
    fi
done

say "Writing $UNIT"
mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=ASL glasses translator (main.py --headless)
After=default.target

[Service]
Type=simple
WorkingDirectory=$REPO
ExecStart=$(sdquote "$PY") $(sdquote "$REPO/main.py") --headless$EXTRA
Environment=PYTHONUNBUFFERED=1
${ENV_LINES}Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
cat "$UNIT"

say "Keeping user services running without a login (linger)"
loginctl enable-linger "${USER:-$(id -un)}"

say "Enabling and starting it"
systemctl --user daemon-reload
systemctl --user enable --now "$NAME"
sleep 3
systemctl --user --no-pager status "$NAME" | head -n 12 || true

cat <<EOF

It now starts at every boot. Useful commands:
  journalctl --user -u $NAME -f       watch what it prints (Ctrl+C to stop watching)
  systemctl --user stop $NAME         free the camera for evaluate.py etc.
  systemctl --user restart $NAME      after a git pull
  bash install_service.sh --uninstall
EOF
