"""
Parchea /Metatrader/start.sh en la imagen gmag11/metatrader5_vnc.

Fixes:
1. RPyC server debe correr en Wine Python (tiene MetaTrader5 instalado),
   no en Linux Python 3.11.
2. numpy debe ser <2 para Wine Python 3.9-32bit.
3. rpyc debe ser ==5.2.3 para compatibilidad con el bot.
"""
import re

STARTSH = "/Metatrader/start.sh"

with open(STARTSH, "r") as f:
    content = f.read()

# --- Fix 1: Correr RPyC server con Wine Python ---
# Original: python3 -m mt5linux --host 0.0.0.0 -p $mt5server_port -w $wine_executable python.exe &
# Nuevo:    wine python.exe -m mt5linux --host 0.0.0.0 -p $mt5server_port &
content = re.sub(
    r"python3 -m mt5linux --host 0\.0\.0\.0 -p \$mt5server_port.*",
    "wine python.exe -m mt5linux --host 0.0.0.0 -p $mt5server_port &",
    content,
)

# --- Fix 2: Inyectar pip install para fijar numpy y rpyc antes de arrancar server ---
fix_line = (
    'show_message "[7/7] Fixing numpy + rpyc versions for Wine Python 3.9..."\n'
    '$wine_executable python -m pip install --no-cache-dir "numpy<2" rpyc==5.2.3 2>/dev/null || true\n'
)
target = 'show_message "[7/7] Starting the mt5linux server..."'
if fix_line not in content:
    content = content.replace(target, fix_line + target)

# --- Fix 3: Pinear versiones en la linea de install original (para fresh installs) ---
content = content.replace(
    "--no-cache-dir rpyc plumbum numpy",
    '--no-cache-dir rpyc==5.2.3 plumbum "numpy<2"',
)

with open(STARTSH, "w") as f:
    f.write(content)

print("[patch_start.py] start.sh patched successfully")
