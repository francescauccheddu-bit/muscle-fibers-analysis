#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per separare e chiudere fibre muscolari in maschere binarie.

Usa watershed per separare fibre che si toccano, poi chiude i contorni aperti.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, segmentation
from scipy import ndimage as ndi


def separate_and_close_fibers(binary_mask, min_fiber_size=500, erosion_size=2):
    """
    Separa le fibre che si toccano e chiude i contorni aperti.

    Args:
        binary_mask: Maschera binaria
        min_fiber_size: Area minima per considerare una fibra
        erosion_size: Dimensione erosione per watershed (default: 2)

    Returns:
        Maschera con fibre separate e contorni chiusi
    """
    print(f"Rimozione rumore (oggetti < {min_fiber_size} pixel)...")
    mask_bool = binary_mask > 0
    cleaned = morphology.remove_small_objects(mask_bool, min_size=min_fiber_size)

    # Calcola la distanza transform
    print("Calcolo distance transform...")
    distance = ndi.distance_transform_edt(cleaned)

    # Trova i picchi locali (centri delle fibre)
    print("Identificazione centri fibre...")
    # Erodi per trovare i centri
    eroded = morphology.erosion(cleaned, morphology.disk(erosion_size))
    markers = measure.label(eroded)

    print(f"Trovati {markers.max()} potenziali centri fibre")

    # Watershed per separare le fibre
    print("Separazione fibre con watershed...")
    labels = segmentation.watershed(-distance, markers, mask=cleaned)

    print(f"Fibre separate: {labels.max()}")

    # Ora chiudi i contorni di ogni fibra individualmente
    print("Chiusura contorni per ogni fibra...")
    result = np.zeros_like(binary_mask, dtype=np.uint8)

    for region in measure.regionprops(labels):
        if region.area < min_fiber_size:
            continue

        # Estrai la singola fibra
        mask_single = labels == region.label

        # Chiudi il contorno usando convex hull
        # Questo garantisce che tutti i "cerchietti" siano chiusi
        filled = morphology.convex_hull_image(mask_single)

        # Aggiungi al risultato
        result[filled] = 255

    return result, labels


def visualize_separation(original, separated_labels, closed, output_dir, base_name):
    """Visualizza la separazione e chiusura."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creazione visualizzazioni...")

    # Colora le fibre separate con colori casuali
    n_fibers = separated_labels.max()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(n_fibers + 1, 3))
    colors[0] = [0, 0, 0]  # Sfondo nero

    colored_separated = colors[separated_labels].astype(np.uint8)

    # Visualizzazione completa
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Originale
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Maschera Originale', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Fibre separate e colorate
    axes[0, 1].imshow(colored_separated)
    axes[0, 1].set_title(f'Fibre Separate ({n_fibers} fibre)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Risultato chiuso
    axes[1, 0].imshow(closed, cmap='gray')
    axes[1, 0].set_title('Contorni Chiusi', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Confronto (rosso=originale, verde=chiuso, giallo=sovrapposizione)
    comparison = np.zeros((*original.shape, 3), dtype=np.uint8)
    comparison[original > 0] = [255, 0, 0]  # Rosso originale
    comparison[closed > 0] = [0, 255, 0]    # Verde chiuso
    # Sovrapposizione diventa gialla (rosso + verde)

    axes[1, 1].imshow(comparison)
    axes[1, 1].set_title('Confronto: Rosso=Originale, Verde=Chiuso, Giallo=Sovrapp.',
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Salva
    vis_path = output_dir / f"{base_name}_watershed_separation.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualizzazione salvata in: {vis_path}")
    plt.close()

    # Salva immagine colorata separata
    colored_path = output_dir / f"{base_name}_separated_colored.png"
    cv2.imwrite(str(colored_path), cv2.cvtColor(colored_separated, cv2.COLOR_RGB2BGR))
    print(f"Fibre separate colorate salvate in: {colored_path}")


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Separa e chiude fibre muscolari in maschere binarie'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Percorso alla maschera binaria'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Directory di output (default: output)'
    )
    parser.add_argument(
        '--min-fiber-size',
        type=int,
        default=500,
        help='Area minima pixel per fibra (default: 500)'
    )
    parser.add_argument(
        '--erosion-size',
        type=int,
        default=2,
        help='Dimensione erosione per watershed (default: 2). Aumenta se fibre troppo vicine'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SEPARAZIONE E CHIUSURA FIBRE MUSCOLARI")
    print("=" * 60)

    # Carica
    print(f"\nCaricamento: {args.input}")
    original = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Errore: Impossibile caricare: {args.input}")
        return

    print(f"Dimensioni: {original.shape}")

    # Processa
    print("\nProcessamento...")
    closed, separated_labels = separate_and_close_fibers(
        original,
        min_fiber_size=args.min_fiber_size,
        erosion_size=args.erosion_size
    )

    # Salva risultato
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input).stem

    output_path = output_dir / f"{base_name}_separated_closed.png"
    cv2.imwrite(str(output_path), closed)
    print(f"\nRisultato salvato in: {output_path}")

    # Visualizza
    visualize_separation(original, separated_labels, closed, args.output, base_name)

    # Statistiche
    n_original = measure.label(original > 0).max()
    n_final = separated_labels.max()

    print("\n" + "=" * 60)
    print("STATISTICHE")
    print("=" * 60)
    print(f"Componenti originali (connesse): {n_original}")
    print(f"Fibre separate identificate: {n_final}")
    print(f"Pixel originali: {np.sum(original > 0):,}")
    print(f"Pixel finali (chiusi): {np.sum(closed > 0):,}")
    print(f"Pixel aggiunti: {np.sum(closed > 0) - np.sum(original > 0):,}")
    print("=" * 60)

    print("\nCOMPLETATO!")


if __name__ == '__main__':
    main()
