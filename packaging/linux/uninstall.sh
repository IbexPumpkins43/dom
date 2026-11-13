#!/bin/sh
# Remove dom (the app and its menu entry). Your chats and models in ~/.local/share/dom are kept;
# delete that folder yourself if you want them gone too.
set -eu
DEST="$HOME/.local/opt/dom"
rm -rf "$DEST"
rm -f "${XDG_DATA_HOME:-$HOME/.local/share}/applications/dom.desktop"
echo "dom has been removed. Your chats and models are still in ~/.local/share/dom."
