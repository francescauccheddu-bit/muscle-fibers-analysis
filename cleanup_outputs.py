#!/usr/bin/env python3
"""
Script per pulire le directory di output di test e spostarle in backup.

Uso:
    python cleanup_outputs.py
"""

import os
import shutil
from pathlib import Path

# Directory da spostare in backup
OUTPUT_DIRS = [
    'output',
    'output_segmentation',
    'output_closed',
    'output_simple_closing',
    'output_multithreshold',
    'output_k15_analysis',
    'output_k21_analysis',
    'test_output',
    'test_closed'
]

BACKUP_DIR = 'backup_output'

def main():
    print("="*80)
    print("PULIZIA OUTPUT DI TEST")
    print("="*80)

    # Crea backup directory
    backup_path = Path(BACKUP_DIR)
    backup_path.mkdir(exist_ok=True)
    print(f"\nBackup directory: {backup_path.absolute()}")

    moved_count = 0
    not_found_count = 0

    for dir_name in OUTPUT_DIRS:
        dir_path = Path(dir_name)

        if dir_path.exists() and dir_path.is_dir():
            dest_path = backup_path / dir_name

            # Se esiste già in backup, rimuovi la vecchia versione
            if dest_path.exists():
                shutil.rmtree(dest_path)

            # Sposta
            shutil.move(str(dir_path), str(dest_path))
            print(f"  ✓ Spostato: {dir_name} → {BACKUP_DIR}/{dir_name}")
            moved_count += 1
        else:
            print(f"  - Non trovato: {dir_name}")
            not_found_count += 1

    print("\n" + "="*80)
    print(f"COMPLETATO")
    print(f"  Directory spostate: {moved_count}")
    print(f"  Directory non trovate: {not_found_count}")
    print("="*80)

    print(f"\nTutte le directory di test sono state spostate in: {BACKUP_DIR}/")
    print("Mantieni solo 'output_final/' come directory di lavoro principale.")


if __name__ == '__main__':
    main()
