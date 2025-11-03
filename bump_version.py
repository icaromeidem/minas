#!/usr/bin/env python3
"""
Script para gerenciar versionamento semântico do MINAS.

Versionamento Semântico (MAJOR.MINOR.PATCH):
- MAJOR (1.x.x): Mudanças significativas/breaking changes
- MINOR (x.1.x): Novas funcionalidades (backward compatible)
- PATCH (x.x.1): Correções de bugs/mudanças pequenas

Uso:
    python bump_version.py major   # 1.0.0 -> 2.0.0
    python bump_version.py minor   # 1.0.0 -> 1.1.0
    python bump_version.py patch   # 1.0.0 -> 1.0.1
"""

import sys
import re
from pathlib import Path

def read_version():
    """Lê a versão atual do arquivo VERSION."""
    version_file = Path(__file__).parent / "VERSION"
    return version_file.read_text().strip()

def write_version(version):
    """Escreve a nova versão no arquivo VERSION."""
    version_file = Path(__file__).parent / "VERSION"
    version_file.write_text(version + "\n")

def update_pyproject_toml(version):
    """Atualiza a versão no pyproject.toml."""
    pyproject_file = Path(__file__).parent / "pyproject.toml"
    content = pyproject_file.read_text()
    
    # Substituir a linha de versão
    new_content = re.sub(
        r'version = "[^"]+"',
        f'version = "{version}"',
        content
    )
    
    pyproject_file.write_text(new_content)

def bump_version(bump_type):
    """
    Incrementa a versão baseado no tipo de mudança.
    
    Args:
        bump_type: 'major', 'minor', ou 'patch'
    """
    current = read_version()
    major, minor, patch = map(int, current.split('.'))
    
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        print(f"Tipo inválido: {bump_type}")
        print("Use: major, minor, ou patch")
        sys.exit(1)
    
    new_version = f"{major}.{minor}.{patch}"
    
    # Atualizar arquivos
    write_version(new_version)
    update_pyproject_toml(new_version)
    
    print(f"✓ Versão atualizada: {current} -> {new_version}")
    print(f"\nPróximos passos:")
    print(f"  1. git add VERSION pyproject.toml")
    print(f"  2. git commit -m 'Bump version to {new_version}'")
    print(f"  3. git tag v{new_version}")
    print(f"  4. git push && git push --tags")
    
    return new_version

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    bump_version(bump_type)
