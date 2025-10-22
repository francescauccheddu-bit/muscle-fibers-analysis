#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per chiudere i contorni aperti in maschere binarie di fibre muscolari.

Questo script prende una maschera binaria esistente e chiude i gap nei contorni
usando operazioni morfologiche.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure


def close_fiber_contours(binary_mask, closing_size=3, fill_holes=True):
    """
    Chiude i contorni aperti nella maschera binaria.

    Args:
        binary_mask: Maschera binaria (0=sfondo, 255=fibre)
        closing_size: Dimensione del kernel per l'operazione di closing (default: 3)
        fill_holes: Se True, riempie anche i buchi interni (default: True)

    Returns:
        Maschera binaria con contorni chiusi
    """
    # Converti in booleano
    mask_bool = binary_mask > 0

    # 1. Operazione di Closing morfologico - chiude gap piccoli
    print(f"Applicazione closing morfologico (kernel size: {closing_size})...")
    closed = morphology.binary_closing(mask_bool, morphology.disk(closing_size))

    # 2. Riempimento buchi (opzionale)
    if fill_holes:
        print("Riempimento buchi interni...")
        closed = morphology.remove_small_holes(closed, area_threshold=100)

    # 3. Pulizia: rimuovi oggetti troppo piccoli
    print("Rimozione oggetti piccoli...")
    closed = morphology.remove_small_objects(closed, min_size=50)

    # Converti in uint8
    result = (closed * 255).astype(np.uint8)

    return result


def analyze_improvements(original, closed):
    """Analizza i miglioramenti apportati."""
    # Conta le componenti connesse
    original_labels = measure.label(original > 0)
    closed_labels = measure.label(closed > 0)

    n_original = original_labels.max()
    n_closed = closed_labels.max()

    # Calcola pixel aggiunti/rimossi
    pixels_added = np.sum((closed > 0) & (original == 0))
    pixels_removed = np.sum((closed == 0) & (original > 0))

    return {
        'original_fibers': n_original,
        'closed_fibers': n_closed,
        'pixels_added': pixels_added,
        'pixels_removed': pixels_removed
    }


def visualize_results(original, closed, output_dir, base_name, stats):
    """Crea visualizzazioni del prima/dopo."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crea immagine delle differenze
    # Verde = pixel aggiunti (contorni chiusi)
    # Rosso = pixel rimossi
    # Bianco = invariato
    diff_image = np.zeros((*original.shape, 3), dtype=np.uint8)

    # Pixel invariati (bianchi)
    both = (original > 0) & (closed > 0)
    diff_image[both] = [255, 255, 255]

    # Pixel aggiunti (verdi) - contorni chiusi
    added = (closed > 0) & (original == 0)
    diff_image[added] = [0, 255, 0]

    # Pixel rimossi (rossi)
    removed = (closed == 0) & (original > 0)
    diff_image[removed] = [255, 0, 0]

    # Visualizzazione
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Originale
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title(f'Maschera Originale\n({stats["original_fibers"]} componenti)',
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Dopo closing
    axes[0, 1].imshow(closed, cmap='gray')
    axes[0, 1].set_title(f'Dopo Closing dei Contorni\n({stats["closed_fibers"]} componenti)',
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Differenze
    axes[1, 0].imshow(diff_image)
    axes[1, 0].set_title(f'Differenze\nVerde=Aggiunti ({stats["pixels_added"]:,}px), '
                        f'Rosso=Rimossi ({stats["pixels_removed"]:,}px)',
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Solo le aggiunte (contorni chiusi)
    only_added = np.zeros_like(diff_image)
    only_added[added] = [0, 255, 0]
    only_added[both] = [100, 100, 100]  # Grigio scuro per contesto
    axes[1, 1].imshow(only_added)
    axes[1, 1].set_title(f'Solo Pixel Aggiunti (verde)\nContorni Chiusi',
                        fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Salva visualizzazione
    vis_path = output_dir / f"{base_name}_closing_comparison.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualizzazione salvata in: {vis_path}")
    plt.close()

    # Salva immagine delle differenze separata
    diff_path = output_dir / f"{base_name}_differences.png"
    cv2.imwrite(str(diff_path), cv2.cvtColor(diff_image, cv2.COLOR_RGB2BGR))
    print(f"Immagine differenze salvata in: {diff_path}")

    # Crea immagine dedicata solo per le chiusure (molto visibile)
    closures_highlight = np.zeros((*original.shape, 3), dtype=np.uint8)

    # Fibre originali in grigio chiaro (per contesto)
    closures_highlight[original > 0] = [180, 180, 180]

    # Chiusure in ROSSO brillante (ben visibile!)
    closures_highlight[added] = [255, 0, 0]

    # Salva immagine chiusure
    closures_path = output_dir / f"{base_name}_closures_only.png"
    cv2.imwrite(str(closures_path), cv2.cvtColor(closures_highlight, cv2.COLOR_RGB2BGR))
    print(f"Immagine solo chiusure salvata in: {closures_path}")

    # Crea anche versione con solo le chiusure su sfondo nero (senza contesto)
    closures_only_black = np.zeros((*original.shape, 3), dtype=np.uint8)
    closures_only_black[added] = [0, 255, 255]  # Ciano brillante

    closures_black_path = output_dir / f"{base_name}_closures_isolated.png"
    cv2.imwrite(str(closures_black_path), cv2.cvtColor(closures_only_black, cv2.COLOR_RGB2BGR))
    print(f"Immagine chiusure isolate salvata in: {closures_black_path}")


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Chiude i contorni aperti in maschere binarie di fibre muscolari'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Percorso alla maschera binaria di input'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Directory di output (default: output)'
    )
    parser.add_argument(
        '--closing-size',
        type=int,
        default=3,
        help='Dimensione kernel per closing morfologico (default: 3). Aumenta per gap più grandi'
    )
    parser.add_argument(
        '--no-fill-holes',
        action='store_true',
        help='Non riempire i buchi interni nelle fibre'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CHIUSURA CONTORNI FIBRE MUSCOLARI")
    print("=" * 60)

    # Carica immagine
    print(f"\nCaricamento: {args.input}")
    original = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Errore: Impossibile caricare l'immagine: {args.input}")
        return

    print(f"Dimensioni immagine: {original.shape}")
    print(f"Range valori: {original.min()} - {original.max()}")

    # Chiudi contorni
    print("\nProcessamento...")
    closed = close_fiber_contours(
        original,
        closing_size=args.closing_size,
        fill_holes=not args.no_fill_holes
    )

    # Analizza miglioramenti
    print("\nAnalisi risultati...")
    stats = analyze_improvements(original, closed)

    # Salva risultato
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input).stem

    output_path = output_dir / f"{base_name}_closed.png"
    cv2.imwrite(str(output_path), closed)
    print(f"Maschera chiusa salvata in: {output_path}")

    # Visualizza risultati
    print("\nCreazione visualizzazioni...")
    visualize_results(original, closed, args.output, base_name, stats)

    # Stampa statistiche
    print("\n" + "=" * 60)
    print("STATISTICHE")
    print("=" * 60)
    print(f"Componenti originali: {stats['original_fibers']}")
    print(f"Componenti dopo closing: {stats['closed_fibers']}")
    print(f"Differenza: {stats['closed_fibers'] - stats['original_fibers']}")
    print(f"\nPixel aggiunti (gap chiusi): {stats['pixels_added']:,}")
    print(f"Pixel rimossi (pulizia): {stats['pixels_removed']:,}")
    print(f"Pixel netti aggiunti: {stats['pixels_added'] - stats['pixels_removed']:,}")
    print("=" * 60)

    print("\nPROCESSAMENTO COMPLETATO!")
    print("=" * 60)


if __name__ == '__main__':
    main()
