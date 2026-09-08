#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
cat > "$ROOT/.venv/bin/veyra" <<WRAP
#!/usr/bin/env bash
exec "$ROOT/.venv/bin/python" -m veyra "\$@"
WRAP
chmod +x "$ROOT/.venv/bin/veyra"
echo "VEYRA environment created at $ROOT/.venv"
echo "Run: $ROOT/.venv/bin/veyra"
echo "For a global command, add $ROOT/.venv/bin to PATH."
