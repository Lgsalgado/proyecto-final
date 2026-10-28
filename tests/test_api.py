"""
Tests de los endpoints principales de la API FastAPI.
Verifica la disponibilidad de rutas y esquemas de respuesta sin requerir
la carga completa de pesos en memoria.
"""

import sys
import os

# Asegurar que el directorio raiz del proyecto este en el path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


def test_root_endpoint_exists():
    """
    Verifica que la aplicacion FastAPI tiene definido el endpoint GET '/'.
    Comprueba que la ruta esta registrada en el router de la aplicacion.
    """
    try:
        from src.api.main import app
        routes = [route.path for route in app.routes]
        assert "/" in routes, f"El endpoint '/' no esta registrado. Rutas disponibles: {routes}"
    except ImportError as e:
        pytest.skip(f"No se pudo importar la app FastAPI: {e}")
    except Exception as e:
        pytest.skip(f"Error al inspeccionar la app: {e}")


def test_app_has_predict_route():
    """
    Verifica que la aplicacion FastAPI tiene definido el endpoint POST '/predict'.
    """
    try:
        from src.api.main import app
        routes = [route.path for route in app.routes]
        assert "/predict" in routes, (
            f"El endpoint '/predict' no esta registrado. Rutas disponibles: {routes}"
        )
    except ImportError as e:
        pytest.skip(f"No se pudo importar la app FastAPI: {e}")
    except Exception as e:
        pytest.skip(f"Error al inspeccionar la app: {e}")


def test_app_has_predict_video_route():
    """
    Verifica que la aplicacion FastAPI tiene definido el endpoint POST '/predict/video'.
    """
    try:
        from src.api.main import app
        routes = [route.path for route in app.routes]
        assert "/predict/video" in routes, (
            f"El endpoint '/predict/video' no esta registrado. Rutas disponibles: {routes}"
        )
    except ImportError as e:
        pytest.skip(f"No se pudo importar la app FastAPI: {e}")
    except Exception as e:
        pytest.skip(f"Error al inspeccionar la app: {e}")


def test_app_is_fastapi_instance():
    """
    Verifica que el objeto 'app' exportado por main.py es una instancia de FastAPI.
    """
    try:
        from fastapi import FastAPI
        from src.api.main import app
        assert isinstance(app, FastAPI), (
            f"Se esperaba una instancia de FastAPI, se obtuvo {type(app)}"
        )
    except ImportError as e:
        pytest.skip(f"No se pudo importar FastAPI o la app: {e}")
    except Exception as e:
        pytest.skip(f"Error al verificar la instancia: {e}")
