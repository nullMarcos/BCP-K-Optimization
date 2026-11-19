"""
Módulo de funciones utilitarias y auxiliares para el proyecto BCP_k.

Contiene herramientas para configuración de entornos y licencias de solvers,
manejo de archivos de salida y cálculo de métricas generales.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import re
from typing import Optional


def _rutas_candidatas(nombre_archivo: str = "gurobi.lic") -> list[Path]:
    """Devuelve rutas candidatas ordenadas por prioridad según el SO."""
    sistema = platform.system().lower()
    home = Path.home()
    cwd = Path.cwd()

    candidatas: list[Path] = [
        cwd / nombre_archivo,
        cwd / "config" / nombre_archivo,
        cwd / "licenses" / nombre_archivo,
        home / nombre_archivo,
        home / "gurobi" / nombre_archivo,
    ]

    if sistema == "linux":
        candidatas.extend([
            Path("/opt/gurobi") / nombre_archivo,
            Path("/usr/local/gurobi") / nombre_archivo,
        ])
    elif sistema == "darwin":  # macOS
        candidatas.extend([
            Path("/Library/gurobi") / nombre_archivo,
            home / "Library" / "gurobi" / nombre_archivo,
            Path("/opt/gurobi") / nombre_archivo,
        ])
    elif sistema == "windows":
        candidatas.extend([
            Path("C:/gurobi") / nombre_archivo,
            Path("C:/gurobi130") / nombre_archivo,
            Path("C:/gurobi120") / nombre_archivo,
            Path("C:/gurobi110") / nombre_archivo,
        ])
        for env_var in ("USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA"):
            base = os.environ.get(env_var)
            if base:
                candidatas.extend([
                    Path(base) / nombre_archivo,
                    Path(base) / "gurobi" / nombre_archivo,
                ])

    # Deduplicar conservando el orden
    vistas: set[Path] = set()
    unicas: list[Path] = []
    for ruta in candidatas:
        resuelta = ruta.expanduser()
        if resuelta not in vistas:
            vistas.add(resuelta)
            unicas.append(resuelta)
    return unicas


def _leer_metadatos_licencia(ruta: Path) -> dict[str, str]:
    """Lee campos no sensibles de gurobi.lic (evitando exponer KEY)."""
    texto = ruta.read_text(encoding="utf-8", errors="ignore")
    metadatos: dict[str, str] = {}
    for campo in ("LICENSEID", "HOSTNAME", "HOSTID", "TYPE", "VERSION"):
        match = re.search(rf"^\s*{campo}\s*=\s*(.+?)\s*$", texto, re.MULTILINE)
        if match:
            metadatos[campo] = match.group(1).strip('"')
    return metadatos


def configurar_licencia(
    ruta_licencia: str | Path | None = None,
    nombre_archivo: str = "gurobi.lic",
    forzar_reemplazo: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Busca y configura automáticamente el archivo gurobi.lic según el SO.

    Args:
        ruta_licencia: Ruta explícita opcional a un archivo .lic.
        nombre_archivo: Nombre del archivo de licencia (por defecto 'gurobi.lic').
        forzar_reemplazo: Si True, sobrescribe GRB_LICENSE_FILE aunque ya exista.
        verbose: Si True, imprime un resumen de la licencia detectada.

    Returns:
        Path resuelto al archivo de licencia configurado.

    Raises:
        FileNotFoundError: Si no se encuentra ningún archivo de licencia válido.
        ValueError: Si el archivo encontrado está vacío o corrupto.
    """
    if "GRB_LICENSE_FILE" in os.environ and not forzar_reemplazo and ruta_licencia is None:
        ruta_actual = Path(os.environ["GRB_LICENSE_FILE"]).expanduser()
        if ruta_actual.is_file():
            if verbose:
                print(f"[Gurobi] Usando GRB_LICENSE_FILE existente: {ruta_actual}")
            return ruta_actual

    licencia_encontrada: Path | None = None

    if ruta_licencia is not None:
        candidata = Path(ruta_licencia).expanduser()
        if not candidata.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo de licencia especificado en: {candidata}"
            )
        licencia_encontrada = candidata.resolve()
    else:
        candidatas: list[Path] = []
        if "GRB_LICENSE_FILE" in os.environ:
            candidatas.append(Path(os.environ["GRB_LICENSE_FILE"]).expanduser())
        candidatas.extend(_rutas_candidatas(nombre_archivo))

        for candidata in candidatas:
            if candidata.is_file():
                licencia_encontrada = candidata.resolve()
                break

        if licencia_encontrada is None:
            rutas_str = "\n  - ".join(str(p) for p in candidatas[:8])
            raise FileNotFoundError(
                f"No se encontró '{nombre_archivo}' en {platform.system()}.\n"
                f"Coloca tu licencia individual en tu carpeta personal (ej. ~/gurobi.lic "
                f"o C:\\Users\\<Usuario>\\gurobi.lic) o pásala con ruta_licencia=...\n"
                f"Rutas revisadas (principales):\n  - {rutas_str}"
            )

    if licencia_encontrada.stat().st_size == 0:
        raise ValueError(f"El archivo de licencia está vacío: {licencia_encontrada}")

    meta = _leer_metadatos_licencia(licencia_encontrada)
    if "KEY" not in licencia_encontrada.read_text(encoding="utf-8", errors="ignore"):
        raise ValueError(
            f"El archivo '{licencia_encontrada}' no parece contener una clave KEY válida."
        )

    # Advertencia preventiva de incompatibilidad de versión de Gurobi
    lic_version = meta.get("VERSION")
    if lic_version:
        try:
            import gurobipy as gp
            grb_major = gp.__version__.split(".")[0]
            if str(lic_version) != grb_major:
                print(
                    f"\n[AVISO GUROBI] Incompatibilidad de versión:\n"
                    f"  -> Tu archivo de licencia es para Gurobi {lic_version}.\n"
                    f"  -> Tu versión instalada de gurobipy es {gp.__version__}.\n"
                    f"  -> Debes instalar una versión compatible con tu licencia:\n"
                    f"     pip install 'gurobipy>={lic_version}.0.0,<{int(lic_version)+1}.0.0'\n"
                )
        except Exception:
            pass

    os.environ["GRB_LICENSE_FILE"] = str(licencia_encontrada)

    if verbose:
        lic_id = meta.get("LICENSEID", "desconocido")
        host = meta.get("HOSTNAME", "no especificado")
        print(
            f"[Gurobi] Licencia configurada ({platform.system()}): "
            f"{licencia_encontrada} | LICENSEID={lic_id} | HOSTNAME={host}"
        )

    return licencia_encontrada
