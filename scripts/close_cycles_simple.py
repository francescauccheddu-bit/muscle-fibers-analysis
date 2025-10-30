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

    Conta il NUMERO di cicli (contorni separati), non i pixel totali.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Trova cicli prima e dopo
    original_filled = ndi.binary_fill_holes(mask > 0).astype(np.uint8) * 255
    closed_filled = ndi.binary_fill_holes(mask_closed > 0).astype(np.uint8) * 255

    cycles_original = cv2.subtract(original_filled, mask)
    cycles_closed = cv2.subtract(closed_filled, mask_closed)

    # Conta NUMERO di cicli (contorni separati), non pixel
    contours_orig, _ = cv2.findContours(cycles_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_after, _ = cv2.findContours(cycles_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra cicli molto piccoli (rumore)
    min_cycle_area = 100
    contours_orig = [c for c in contours_orig if cv2.contourArea(c) >= min_cycle_area]
    contours_after = [c for c in contours_after if cv2.contourArea(c) >= min_cycle_area]

    n_cycles_original = len(contours_orig)
    n_cycles_closed_after = len(contours_after)
    n_pixels_added = np.sum(cv2.subtract(mask_closed, mask) > 0)

    # Pixel di ciclo
    cycle_pixels_original = np.sum(cycles_original > 0)
    cycle_pixels_after = np.sum(cycles_closed > 0)

    improvement = n_cycles_original - n_cycles_closed_after
    improvement_pct = (improvement / n_cycles_original * 100) if n_cycles_original > 0 else 0

    return {
        'kernel_size': kernel_size,
        'mask_closed': mask_closed,
        'n_cycles_original': int(n_cycles_original),
        'n_cycles_after': int(n_cycles_closed_after),
        'cycles_closed': int(improvement),
        'improvement_percentage': float(improvement_pct),
        'pixels_added': int(n_pixels_added),
        'cycle_pixels_original': int(cycle_pixels_original),
        'cycle_pixels_after': int(cycle_pixels_after)
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
    print(f"\n{'Kernel':<10} {'#Cicli Orig':<15} {'#Cicli Dopo':<15} {'Cicli Chiusi':<15} {'% Miglior.':<12} {'Pixel+':<12}")
    print("-" * 80)

    results = []
    for kernel_size in kernel_sizes:
        result = close_and_measure(mask, kernel_size)

        # Salva maschera prima di rimuoverla dal dict
        mask_path = output_dir / f"mask_closed_k{kernel_size}.png"
        cv2.imwrite(str(mask_path), result['mask_closed'])

        # Rimuovi mask_closed prima di aggiungerlo ai results (non è JSON serializable)
        mask_closed = result.pop('mask_closed')
        results.append(result)

        print(f"{result['kernel_size']:<10} "
              f"{result['n_cycles_original']:>14,} "
              f"{result['n_cycles_after']:>14,} "
              f"{result['cycles_closed']:>14,} "
              f"{result['improvement_percentage']:>11.1f}% "
              f"{result['pixels_added']:>11,}")

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
    axes[0, 1].set_title(f'Cicli Aperti: {results[0]["n_cycles_original"]:,} contorni', fontweight='bold')
    axes[0, 1].axis('off')

    # Righe successive: kernel testati
    for plot_idx, result_idx in enumerate(indices, start=1):
        result = results[result_idx]

        # Ricarica maschera chiusa dal file
        mask_closed = cv2.imread(str(output_dir / f"mask_closed_k{result['kernel_size']}.png"), cv2.IMREAD_GRAYSCALE)

        # Maschera chiusa
        axes[plot_idx, 0].imshow(mask_closed, cmap='gray')
        axes[plot_idx, 0].set_title(
            f"Kernel {result['kernel_size']}×{result['kernel_size']} | "
            f"+{result['pixels_added']:,} pixel",
            fontsize=10
        )
        axes[plot_idx, 0].axis('off')

        # Cicli residui
        cycles_after = cv2.subtract(
            ndi.binary_fill_holes(mask_closed > 0).astype(np.uint8) * 255,
            mask_closed
        )
        axes[plot_idx, 1].imshow(cycles_after, cmap='hot')
        axes[plot_idx, 1].set_title(
            f"Cicli Residui: {result['n_cycles_after']:,} contorni | "
            f"Chiusi: {result['cycles_closed']} ({result['improvement_percentage']:.1f}%)",
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
