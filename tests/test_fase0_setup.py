"""
VALIDACION FASE 0 - Setup del Proyecto
Verifica que toda la estructura y dependencias estan correctas.
Ejecutar: python -m pytest tests/test_fase0_setup.py -v
"""
import os
import sys
import importlib
from pathlib import Path

# Agregar el directorio raiz al path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

BASE_DIR = ROOT_DIR / "forex_bot"


class TestEstructuraCarpetas:
    """Verifica que todas las carpetas existen."""

    def test_forex_bot_existe(self):
        assert BASE_DIR.is_dir(), "forex_bot/ no existe"

    def test_config_dir(self):
        assert (BASE_DIR / "config").is_dir(), "config/ no existe"

    def test_data_dir(self):
        assert (BASE_DIR / "data").is_dir(), "data/ no existe"

    def test_models_dir(self):
        assert (BASE_DIR / "models").is_dir(), "models/ no existe"

    def test_models_saved_dir(self):
        assert (BASE_DIR / "models" / "saved").is_dir(), "models/saved/ no existe"

    def test_strategy_dir(self):
        assert (BASE_DIR / "strategy").is_dir(), "strategy/ no existe"

    def test_execution_dir(self):
        assert (BASE_DIR / "execution").is_dir(), "execution/ no existe"

    def test_backtesting_dir(self):
        assert (BASE_DIR / "backtesting").is_dir(), "backtesting/ no existe"

    def test_utils_dir(self):
        assert (BASE_DIR / "utils").is_dir(), "utils/ no existe"

    def test_logs_dir(self):
        assert (BASE_DIR / "logs").is_dir(), "logs/ no existe"


class TestInitFiles:
    """Verifica que todos los __init__.py existen."""

    packages = [
        "forex_bot",
        "forex_bot/config",
        "forex_bot/data",
        "forex_bot/models",
        "forex_bot/strategy",
        "forex_bot/execution",
        "forex_bot/backtesting",
        "forex_bot/utils",
    ]

    def test_init_files_existen(self):
        for pkg in self.packages:
            init_path = ROOT_DIR / pkg / "__init__.py"
            assert init_path.is_file(), f"__init__.py falta en {pkg}/"


class TestArchivosBase:
    """Verifica que archivos clave del proyecto existen."""

    def test_requirements_txt(self):
        assert (ROOT_DIR / "requirements.txt").is_file(), "requirements.txt no existe"

    def test_gitignore(self):
        assert (ROOT_DIR / ".gitignore").is_file(), ".gitignore no existe"

    def test_plan_maestro(self):
        assert (ROOT_DIR / "PLAN_MAESTRO.md").is_file(), "PLAN_MAESTRO.md no existe"

    def test_ideas_futuras(self):
        assert (ROOT_DIR / "IDEAS_FUTURAS.md").is_file(), "IDEAS_FUTURAS.md no existe"

    def test_mt5_config_example(self):
        assert (BASE_DIR / "config" / "mt5_config.example.py").is_file(), \
            "mt5_config.example.py no existe"


class TestGitignore:
    """Verifica que .gitignore tiene las exclusiones criticas."""

    def setup_method(self):
        with open(ROOT_DIR / ".gitignore", "r") as f:
            self.content = f.read()

    def test_excluye_pycache(self):
        assert "__pycache__" in self.content

    def test_excluye_venv(self):
        assert ".venv" in self.content or "venv" in self.content

    def test_excluye_mt5_config(self):
        assert "mt5_config.py" in self.content, \
            "CRITICO: mt5_config.py debe estar en .gitignore para proteger credenciales"

    def test_excluye_modelos(self):
        assert ".pkl" in self.content or ".joblib" in self.content

    def test_excluye_logs(self):
        assert "log" in self.content.lower()


class TestDependenciasInstaladas:
    """Verifica que las dependencias criticas se pueden importar."""

    def test_pandas(self):
        import pandas
        assert pandas.__version__ is not None

    def test_numpy(self):
        import numpy
        assert numpy.__version__ is not None

    def test_sklearn(self):
        import sklearn
        assert sklearn.__version__ is not None

    def test_xgboost(self):
        import xgboost
        assert xgboost.__version__ is not None

    def test_joblib(self):
        import joblib
        assert joblib.__version__ is not None

    def test_matplotlib(self):
        import matplotlib
        assert matplotlib.__version__ is not None

    def test_pandas_ta(self):
        import pandas_ta
        assert pandas_ta.version is not None

    def test_metatrader5(self):
        """MT5 puede no estar instalado en CI, pero debe estar localmente."""
        try:
            import MetaTrader5
            assert True
        except ImportError:
            # Aceptable si no esta en Windows o no tiene MT5
            import platform
            if platform.system() == "Windows":
                assert False, "MetaTrader5 no instalado (requerido en Windows)"
            else:
                assert True, "MetaTrader5 solo disponible en Windows"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
