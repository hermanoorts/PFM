import sys
sys.setrecursionlimit(10000)
from cx_Freeze import setup, Executable

# Ejecutable
executables = [Executable("main.py", base=None)]

# Opciones
build_exe_options = {"packages": ["tkinter", "pandas", "numpy", "sklearn", "joblib"], "include_files": [], "excludes": []}

# Configuración
setup(
    name="PFM_App",
    version="0.1",
    description="Mejor opción de envío",
    options={"build_exe": build_exe_options},
    executables=executables
)
