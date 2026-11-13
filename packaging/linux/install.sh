#!/bin/sh
# Install dom for the current user: the app goes to ~/.local/opt/dom and a menu entry is added.
# Your chats and models live in ~/.local/share/dom and are not touched.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.local/opt/dom"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

rm -rf "$DEST"
mkdir -p "$DEST" "$APPS"
cp -a "$HERE/dom/." "$DEST/"
sed "s|@DEST@|$DEST|g" "$HERE/dom.desktop" > "$APPS/dom.desktop"
chmod +x "$DEST/dom"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" || true
echo "dom is installed. Start it from your applications menu, or run: $DEST/dom"
