#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sovrappone i centroidi delle fibre sull'immagine fluorescenza originale.

Uso:
    python scripts/overlay_centroids_on_fluorescence.py \
        --fluorescence data/laminina_originale.png \
        --mask output_simple_closing/mask_closed_k15.png \
        --output output_k15_analysis/laminina_with_centroids.png \
        --dot-radius 5
"""

import argparse
import numpy as np
import cv2
from scipy import ndimage as ndi


def find_closed_cycles(mask, min_area=1000):
    """
    Identifica cicli chiusi e calcola centroidi.
    """
    print("Identificazione cicli chiusi...")

    # Binary fill holes
    mask_bool = mask > 0
    filled = ndi.binary_fill_holes(mask_bool).astype(np.uint8) * 255

    # Cicli = filled - mask
    cycles = cv2.subtract(filled, mask)

    # Find contours
    contours, _ = cv2.findContours(cycles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"  Trovati {len(contours)} contorni")

    # Filtra per area ed escludi il più grande (background)
    valid_contours = []
    areas = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid_contours.append(contour)
            areas.append(area)

    # Escludi il più grande
    if len(areas) > 0:
        max_area_idx = np.argmax(areas)
        print(f"  Escluso ciclo più grande: {areas[max_area_idx]:.0f} px²")
        valid_contours.pop(max_area_idx)
        areas.pop(max_area_idx)

    # Calcola centroidi
    centroids = []
    for contour in valid_contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            centroids.append((int(cy), int(cx)))

    print(f"  Cicli validi con centroidi: {len(centroids)}")

    return centroids


def create_overlay(fluorescence, centroids, dot_radius=5, dot_color=(0, 0, 255)):
    """
    Crea overlay con pallini rossi sui centroidi.
    """
    print("Creazione overlay...")

    # Converti fluorescenza a RGB se necessario
    if len(fluorescence.shape) == 2:
        overlay = cv2.cvtColor(fluorescence, cv2.COLOR_GRAY2RGB)
    else:
        overlay = fluorescence.copy()

    # Normalizza se 16-bit
    if overlay.dtype == np.uint16:
        overlay = (overlay / 256).astype(np.uint8)

    # Disegna pallini
    for cy, cx in centroids:
        cv2.circle(overlay, (cx, cy), dot_radius, dot_color, -1)

    print(f"  Disegnati {len(centroids)} pallini rossi (radius={dot_radius})")

    return overlay


def main():
    parser = argparse.ArgumentParser(
        description='Sovrappone centroidi fibre su immagine fluorescenza'
    )
    parser.add_argument('--fluorescence', type=str, required=True,
                       help='Immagine fluorescenza laminina originale')
    parser.add_argument('--mask', type=str, required=True,
                       help='Maschera chiusa (es. mask_closed_k15.png)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path file output')
    parser.add_argument('--dot-radius', type=int, default=5,
                       help='Raggio pallini in pixel (default: 5)')
    parser.add_argument('--min-cycle-area', type=int, default=1000,
                       help='Area minima cicli (default: 1000)')

    args = parser.parse_args()

    print("="*80)
    print("OVERLAY CENTROIDI SU FLUORESCENZA")
    print("="*80)

    # Carica immagini
    print(f"\nCaricamento fluorescenza: {args.fluorescence}")
    fluorescence = cv2.imread(args.fluorescence, cv2.IMREAD_UNCHANGED)
    if fluorescence is None:
        raise ValueError(f"Impossibile caricare: {args.fluorescence}")
    print(f"  Dimensioni: {fluorescence.shape}, dtype: {fluorescence.dtype}")

    print(f"\nCaricamento maschera: {args.mask}")
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossibile caricare: {args.mask}")
    print(f"  Dimensioni: {mask.shape}")

    # Trova centroidi
    centroids = find_closed_cycles(mask, min_area=args.min_cycle_area)

    # Crea overlay
    overlay = create_overlay(fluorescence, centroids,
                            dot_radius=args.dot_radius,
                            dot_color=(0, 0, 255))  # BGR: rosso

    # Salva
    print(f"\nSalvataggio: {args.output}")
    cv2.imwrite(args.output, overlay)
    print(f"  Salvato con successo!")

    print("\n" + "="*80)
    print(f"COMPLETATO: {len(centroids)} fibre visualizzate")
    print("="*80)


if __name__ == '__main__':
    main()
