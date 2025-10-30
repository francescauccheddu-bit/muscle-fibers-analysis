#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chiusura cicli semplice con morphological closing progressivo.

Approccio diretto:
- Applica morphological closing con kernel crescente
- Mostra statistiche per ogni dimensione kernel
- L'utente può scegliere quale dimensione funziona meglio

Uso:
    python scripts/close_cycles_simple.py \
        --mask data/Maschera.png \
        --output output_simple_closing
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
import json


def close_and_measure(mask, kernel_size):
    """
    Applica morphological closing e misura i cicli chiusi.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Misura cicli prima e dopo
    original_filled = ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255
    closed_filled = ndi.binary_fill_holes(mask_closed > 0).astype(np.uint8) * 255

    cycles_original = cv2.subtract(original_filled, mask)
    cycles_closed = cv2.subtract(closed_filled, mask_closed)

    n_cycles_original = np.sum(cycles_original > 0)
    n_cycles_closed = np.sum(cycles_closed > 0)
    n_pixels_added = np.sum(cv2.subtract(mask_closed, mask) > 0)

    improvement = n_cycles_original - n_cycles_closed
    improvement_pct = (improvement / n_cycles_original * 100) if n_cycles_original > 0 else 0

    return {
        'kernel_size': kernel_size,
        'mask_closed': mask_closed,
        'cycles_original': int(n_cycles_original),
        'cycles_closed': int(n_cycles_closed),
        'cycles_improvement': int(improvement),
        'improvement_percentage': float(improvement_pct),
        'pixels_added': int(n_pixels_added)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Chiusura cicli con morphological closing progressivo'
    )
    parser.add_argument('--mask', type=str, required=True, help='Maschera di input')
    parser.add_argument('--output', type=str, default='output_simple', help='Directory output')
    parser.add_argument(
        '--kernel-sizes',
        type=str,
        default='3,5,7,9,11,15,21',
        help='Dimensioni kernel da testare (es: 3,5,7,9,11)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CHIUSURA CICLI - MORPHOLOGICAL CLOSING PROGRESSIVO")
    print("=" * 80)

    # Carica maschera
    print(f"\nCaricamento maschera: {args.mask}")
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossibile caricare: {args.mask}")
    print(f"  Dimensioni: {mask.shape}")

    # Crea output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse kernel sizes
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    print(f"\nKernel sizes da testare: {kernel_sizes}")

    # Test ogni kernel
    print(f"\n{'Kernel':<10} {'Cicli Orig':<15} {'Cicli Dopo':<15} {'Miglioramento':<15} {'% Miglior.':<12} {'Pixel+':<12}")
    print("-" * 80)

    results = []
    for kernel_size in kernel_sizes:
        result = close_and_measure(mask, kernel_size)
        results.append(result)

        print(f"{result['kernel_size']:<10} "
              f"{result['cycles_original']:>14,} "
              f"{result['cycles_closed']:>14,} "
              f"{result['cycles_improvement']:>14,} "
              f"{result['improvement_percentage']:>11.1f}% "
              f"{result['pixels_added']:>11,}")

        # Salva maschera
        mask_path = output_dir / f"mask_closed_k{kernel_size}.png"
        cv2.imwrite(str(mask_path), result['mask_closed'])

    # Trova migliore
    best = max(results, key=lambda x: x['improvement_percentage'])
    print(f"\n{'='*80}")
    print(f"MIGLIORE: Kernel {best['kernel_size']} → {best['improvement_percentage']:.1f}% miglioramento")
    print(f"{'='*80}")

    # Salva statistiche
    stats_path = output_dir / 'closing_comparison.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'results': results,
            'best_kernel': best['kernel_size'],
            'best_improvement_pct': best['improvement_percentage']
        }, f, indent=2)
    print(f"\nStatistiche salvate: {stats_path}")

    # Crea visualizzazione comparativa
    print(f"\nCreazione visualizzazione comparativa...")

    # Prendi 3 kernel rappresentativi: piccolo, medio, grande
    if len(kernel_sizes) >= 3:
        indices = [0, len(kernel_sizes)//2, -1]
    else:
        indices = list(range(len(kernel_sizes)))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(indices) + 1, 2, figsize=(12, 4 * (len(indices) + 1)))

    # Riga 0: Originale
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Maschera Originale', fontweight='bold')
    axes[0, 0].axis('off')

    cycles_orig = cv2.subtract(
        ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255,
        mask
    )
    axes[0, 1].imshow(cycles_orig, cmap='hot')
    axes[0, 1].set_title(f'Cicli Aperti: {results[0]["cycles_original"]:,} pixel', fontweight='bold')
    axes[0, 1].axis('off')

    # Righe successive: kernel testati
    for plot_idx, result_idx in enumerate(indices, start=1):
        result = results[result_idx]

        # Maschera chiusa
        axes[plot_idx, 0].imshow(result['mask_closed'], cmap='gray')
        axes[plot_idx, 0].set_title(
            f"Kernel {result['kernel_size']}×{result['kernel_size']} | "
            f"+{result['pixels_added']:,} pixel",
            fontsize=10
        )
        axes[plot_idx, 0].axis('off')

        # Cicli residui
        cycles_after = cv2.subtract(
            ndi.binary_fill_holes(result['mask_closed'] > 0).astype(np.uint8) * 255,
            result['mask_closed']
        )
        axes[plot_idx, 1].imshow(cycles_after, cmap='hot')
        axes[plot_idx, 1].set_title(
            f"Cicli Residui: {result['cycles_closed']:,} pixel | "
            f"Miglioramento: {result['improvement_percentage']:.1f}%",
            fontsize=10
        )
        axes[plot_idx, 1].axis('off')

    plt.tight_layout()
    vis_path = output_dir / 'closing_comparison.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"  Salvato: {vis_path}")
    plt.close()

    print(f"\n{'='*80}")
    print(f"COMPLETATO!")
    print(f"Usa la maschera migliore: {output_dir / f'mask_closed_k{best['kernel_size']}.png'}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
