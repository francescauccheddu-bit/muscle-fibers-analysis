#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per chiusura intelligente dei cicli usando informazioni di intensità.

STEP 2: Dopo aver ottenuto una maschera iniziale (da segment_laminina.py),
        usa multi-threshold sull'immagine originale per chiudere i cicli aperti.

Approccio:
1. Identifica cicli aperti nella maschera
2. Per ogni gap, analizza intensità nell'immagine originale
3. Usa soglie multiple per trovare connessioni plausibili
4. Chiudi cicli dove l'intensità supporta la connessione

Uso:
    python scripts/close_cycles_multithreshold.py \
        --mask data/mask_initial.png \
        --fluorescence data/laminina_originale.png \
        --output output_closed
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage as ndi


def find_gap_regions(mask, skeleton):
    """
    Identifica le regioni di gap dove i cicli non sono chiusi.

    Returns:
        gap_mask: Maschera dei gap potenziali (regioni vicine ma non connesse)
    """
    print("Identificazione regioni di gap...")

    # Trova cicli potenziali (filled - original mask)
    mask_bool = mask > 0
    filled = ndi.binary_fill_holes(mask_bool).astype(np.uint8) * 255
    potential_cycles = cv2.subtract(filled, mask)

    # Dilata leggermente la maschera originale per trovare regioni "vicine"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    # Gap = aree dentro potenziali cicli che sono vicine alla maschera
    gap_mask = cv2.bitwise_and(potential_cycles, mask_dilated)

    print(f"  Gap pixel trovati: {np.sum(gap_mask > 0):,}")

    return gap_mask, potential_cycles


def analyze_intensity_in_gaps(fluorescence, gap_mask, n_thresholds=5):
    """
    Analizza l'intensità dell'immagine fluorescenza nelle regioni di gap.
    Usa multiple soglie per identificare connessioni plausibili.

    Args:
        fluorescence: Immagine originale di fluorescenza
        gap_mask: Maschera delle regioni di gap
        n_thresholds: Numero di soglie da testare

    Returns:
        Dict con maschere per ogni soglia
    """
    print(f"Analisi intensità con {n_thresholds} soglie...")

    # Normalizza a 8-bit se necessario
    if fluorescence.dtype == np.uint16:
        fluor_8bit = (fluorescence / 256).astype(np.uint8)
    else:
        fluor_8bit = fluorescence

    # Estrai valori di intensità solo nelle regioni di gap
    gap_intensities = fluor_8bit[gap_mask > 0]

    if len(gap_intensities) == 0:
        print("  Nessuna regione di gap trovata!")
        return {}

    # Calcola percentili per definire soglie
    percentiles = np.linspace(10, 90, n_thresholds)
    thresholds = [np.percentile(gap_intensities, p) for p in percentiles]

    print(f"  Range intensità in gap: {gap_intensities.min()} - {gap_intensities.max()}")
    print(f"  Soglie: {[int(t) for t in thresholds]}")

    # Per ogni soglia, crea una maschera delle connessioni plausibili
    threshold_masks = {}

    for i, (percentile, threshold) in enumerate(zip(percentiles, thresholds)):
        # Applica soglia all'immagine originale
        _, binary = cv2.threshold(fluor_8bit, threshold, 255, cv2.THRESH_BINARY)

        # Mantieni solo i pixel nei gap
        connection_mask = cv2.bitwise_and(binary, gap_mask)

        n_pixels = np.sum(connection_mask > 0)
        threshold_masks[f'threshold_{i}_p{int(percentile)}'] = {
            'mask': connection_mask,
            'threshold': threshold,
            'percentile': percentile,
            'n_pixels': n_pixels
        }

        print(f"  Soglia {i} (p={int(percentile)}, t={int(threshold)}): {n_pixels:,} pixel connessi")

    return threshold_masks


def progressive_closing(mask, threshold_masks, skeleton):
    """
    Chiude i cicli progressivamente usando le maschere multi-threshold.

    Strategia:
    - Inizia con soglie alte (più conservativo)
    - Aggiungi connessioni solo se chiudono effettivamente un ciclo
    - Evita di aggiungere rumore o connessioni spurie
    """
    print("Chiusura progressiva dei cicli...")

    mask_closed = mask.copy()
    cycles_closed_count = 0

    # Ordina soglie dalla più alta (più conservativa) alla più bassa
    sorted_thresholds = sorted(
        threshold_masks.items(),
        key=lambda x: x[1]['threshold'],
        reverse=True
    )

    for threshold_name, threshold_data in sorted_thresholds:
        connection_mask = threshold_data['mask']

        # Aggiungi connessioni
        mask_with_connections = cv2.bitwise_or(mask_closed, connection_mask)

        # Verifica quanti nuovi cicli vengono chiusi
        before_bool = mask_closed > 0
        after_bool = mask_with_connections > 0

        filled_before = ndi.binary_fill_holes(before_bool)
        filled_after = ndi.binary_fill_holes(after_bool)

        new_filled = cv2.subtract(
            (filled_after * 255).astype(np.uint8),
            (filled_before * 255).astype(np.uint8)
        )

        new_filled_pixels = np.sum(new_filled > 0)

        if new_filled_pixels > 0:
            # Cicli nuovi chiusi! Mantieni queste connessioni
            mask_closed = mask_with_connections
            cycles_closed_count += 1
            print(f"  {threshold_name}: +{new_filled_pixels:,} pixel chiusi")
        else:
            print(f"  {threshold_name}: nessun nuovo ciclo chiuso")

    print(f"Totale iterazioni che hanno chiuso cicli: {cycles_closed_count}/{len(sorted_thresholds)}")

    return mask_closed


def visualize_multithreshold_results(original_fluor, original_mask, closed_mask,
                                     gap_mask, threshold_masks, output_path):
    """
    Visualizza i risultati della chiusura multi-threshold.
    """
    print("Creazione visualizzazione...")

    n_thresholds = len(threshold_masks)
    n_rows = 3 + min(3, n_thresholds)  # Max 3 threshold previews

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 3.5))

    # Riga 1: Originali
    axes[0, 0].imshow(original_fluor, cmap='gray')
    axes[0, 0].set_title('Fluorescenza Originale', fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(original_mask, cmap='gray')
    axes[0, 1].set_title('Maschera Iniziale', fontweight='bold')
    axes[0, 1].axis('off')

    # Riga 2: Gap e risultato finale
    axes[1, 0].imshow(gap_mask, cmap='hot')
    axes[1, 0].set_title('Regioni di Gap Identificate', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(closed_mask, cmap='gray')
    axes[1, 1].set_title('Maschera Chiusa (Multi-Threshold)', fontweight='bold')
    axes[1, 1].axis('off')

    # Riga 3: Confronto prima/dopo
    # Overlay prima
    overlay_before = cv2.cvtColor(original_fluor, cv2.COLOR_GRAY2RGB)
    overlay_before[original_mask > 0] = [0, 255, 0]
    axes[2, 0].imshow(overlay_before)
    axes[2, 0].set_title('Prima: Fluorescenza + Maschera (Verde)', fontweight='bold')
    axes[2, 0].axis('off')

    # Overlay dopo
    overlay_after = cv2.cvtColor(original_fluor, cv2.COLOR_GRAY2RGB)
    overlay_after[closed_mask > 0] = [255, 255, 0]
    overlay_after[original_mask > 0] = [0, 255, 0]
    axes[2, 1].imshow(overlay_after)
    axes[2, 1].set_title('Dopo: Verde=Originale, Giallo=Chiuso', fontweight='bold')
    axes[2, 1].axis('off')

    # Righe successive: Preview di alcune soglie
    sorted_items = sorted(threshold_masks.items(),
                         key=lambda x: x[1]['threshold'],
                         reverse=True)

    for idx in range(min(3, n_thresholds)):
        row = 3 + idx
        threshold_name, threshold_data = sorted_items[idx]

        # Soglia sul fluorescenza
        _, binary = cv2.threshold(original_fluor, threshold_data['threshold'], 255, cv2.THRESH_BINARY)
        axes[row, 0].imshow(binary, cmap='gray')
        axes[row, 0].set_title(
            f"Soglia {idx}: t={int(threshold_data['threshold'])} "
            f"(p={int(threshold_data['percentile'])})",
            fontsize=10
        )
        axes[row, 0].axis('off')

        # Connessioni in gap
        axes[row, 1].imshow(threshold_data['mask'], cmap='hot')
        axes[row, 1].set_title(
            f"Connessioni Gap: {threshold_data['n_pixels']:,} pixel",
            fontsize=10
        )
        axes[row, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Salvato: {output_path}")
    plt.close()


def compute_closing_statistics(original_mask, closed_mask):
    """
    Calcola statistiche sulla chiusura dei cicli.
    """
    print("\nStatistiche chiusura:")

    # Pixel aggiunti
    new_pixels = cv2.subtract(closed_mask, original_mask)
    n_new_pixels = np.sum(new_pixels > 0)

    # Cicli prima e dopo
    original_filled = ndi.binary_fill_holes(original_mask > 0).astype(np.uint8) * 255
    closed_filled = ndi.binary_fill_holes(closed_mask > 0).astype(np.uint8) * 255

    cycles_original = cv2.subtract(original_filled, original_mask)
    cycles_closed = cv2.subtract(closed_filled, closed_mask)

    n_cycles_original = np.sum(cycles_original > 0)
    n_cycles_closed = np.sum(cycles_closed > 0)

    print(f"  Pixel aggiunti: {n_new_pixels:,}")
    print(f"  Pixel cicli PRIMA: {n_cycles_original:,}")
    print(f"  Pixel cicli DOPO: {n_cycles_closed:,}")
    print(f"  Pixel cicli CHIUSI: {n_cycles_original - n_cycles_closed:,} ({((n_cycles_original - n_cycles_closed) / n_cycles_original * 100):.1f}%)")

    return {
        'new_pixels': int(n_new_pixels),
        'cycles_original': int(n_cycles_original),
        'cycles_closed': int(n_cycles_closed),
        'cycles_improvement': int(n_cycles_original - n_cycles_closed),
        'improvement_percentage': float((n_cycles_original - n_cycles_closed) / n_cycles_original * 100) if n_cycles_original > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(
        description='Chiusura intelligente cicli con multi-threshold'
    )
    parser.add_argument(
        '--mask',
        type=str,
        required=True,
        help='Maschera iniziale (output da segment_laminina.py)'
    )
    parser.add_argument(
        '--fluorescence',
        type=str,
        required=True,
        help='Immagine originale fluorescenza laminina'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_closed',
        help='Directory di output (default: output_closed)'
    )
    parser.add_argument(
        '--n-thresholds',
        type=int,
        default=5,
        help='Numero di soglie da testare (default: 5)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CHIUSURA CICLI CON MULTI-THRESHOLD")
    print("=" * 80)

    # Crea directory output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carica immagini
    print(f"\n1. CARICAMENTO IMMAGINI")
    print("-" * 80)

    print(f"Caricamento maschera: {args.mask}")
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossibile caricare: {args.mask}")
    print(f"  Dimensioni: {mask.shape}")

    print(f"Caricamento fluorescenza: {args.fluorescence}")
    fluorescence = cv2.imread(args.fluorescence, cv2.IMREAD_GRAYSCALE)
    if fluorescence is None:
        raise ValueError(f"Impossibile caricare: {args.fluorescence}")
    print(f"  Dimensioni: {fluorescence.shape}")

    if mask.shape != fluorescence.shape:
        raise ValueError(f"Dimensioni diverse! Mask: {mask.shape}, Fluor: {fluorescence.shape}")

    # Skeleton per analisi
    skeleton = morphology.skeletonize(mask > 0).astype(np.uint8) * 255

    # STEP 1: Trova gap
    print(f"\n2. IDENTIFICAZIONE GAP")
    print("-" * 80)
    gap_mask, potential_cycles = find_gap_regions(mask, skeleton)

    # STEP 2: Analizza intensità con multi-threshold
    print(f"\n3. ANALISI MULTI-THRESHOLD")
    print("-" * 80)
    threshold_masks = analyze_intensity_in_gaps(fluorescence, gap_mask, args.n_thresholds)

    if not threshold_masks:
        print("Nessun gap trovato o nessuna intensità valida. Fine.")
        return

    # STEP 3: Chiusura progressiva
    print(f"\n4. CHIUSURA PROGRESSIVA")
    print("-" * 80)
    mask_closed = progressive_closing(mask, threshold_masks, skeleton)

    # STEP 4: Statistiche
    print(f"\n5. STATISTICHE")
    print("-" * 80)
    stats = compute_closing_statistics(mask, mask_closed)

    # STEP 5: Salvataggio
    print(f"\n6. SALVATAGGIO")
    print("-" * 80)

    cv2.imwrite(str(output_dir / 'mask_closed.png'), mask_closed)
    print(f"  Salvato: {output_dir / 'mask_closed.png'}")

    cv2.imwrite(str(output_dir / 'gap_regions.png'), gap_mask)
    print(f"  Salvato: {output_dir / 'gap_regions.png'}")

    # Visualizzazione
    vis_path = output_dir / 'multithreshold_visualization.png'
    visualize_multithreshold_results(
        fluorescence, mask, mask_closed,
        gap_mask, threshold_masks, vis_path
    )

    # Salva statistiche
    import json
    stats_path = output_dir / 'closing_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Salvato: {stats_path}")

    print(f"\n" + "=" * 80)
    print(f"COMPLETATO!")
    print(f"Miglioramento cicli: {stats['cycles_improvement']:,} pixel ({stats['improvement_percentage']:.1f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
