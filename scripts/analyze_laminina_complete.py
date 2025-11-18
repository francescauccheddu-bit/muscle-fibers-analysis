#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline completa per analisi fibre muscolari da immagine fluorescenza laminina.

STEP 1: Segmentazione automatica (adaptive threshold)
STEP 2: Morphological closing (default K15)
STEP 3: Identificazione cicli chiusi
STEP 4: Calcolo statistiche ed esportazione

Output:
  === VISUALIZZAZIONI FASI (con overlay trasparente) ===
  - step1_segmentation_overlay.png    # Contorni verdi segmentazione su fluorescenza 50% trasparente
  - step2_closing_overlay.png         # Contorni arancioni closing + gap cyan su fluorescenza trasparente
  - step3_cycles_overlay.png          # Contorni blu cicli chiusi + centroidi rossi su trasparente
  - step4_skeleton_overlay.png        # Skeleton blu + centroidi rossi su trasparente

  === VISUALIZZAZIONI FINALI ===
  - laminina_with_centroids.png       # Fluorescenza + gap cyan + centroidi rossi
  - skeleton_with_centroids.png       # Cicli chiusi bianchi + gap cyan + centroidi rossi
  - closing_gaps_visualization.png    # Gap in cyan su maschera grigia
  - area_distribution.png             # Istogramma distribuzione aree

  === DATI ===
  - fibers_statistics.csv             # Statistiche per ogni fibra
  - summary_statistics.csv            # Statistiche sommarie
  - metadata.json                     # Metadati analisi

Uso:
    python scripts/analyze_laminina_complete.py \
        --input data/laminina_originale.png \
        --output output_final \
        --kernel-size 15 \
        --min-fiber-area 1000 \
        --dot-radius 5 \
        --pixel-size 0.41026
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology
import json


def segment_adaptive_threshold(image, block_size=101, C=2):
    """
    Segmentazione usando adaptive thresholding.
    """
    print("  Metodo: Adaptive Threshold")
    print(f"    Block size: {block_size}, C: {C}")

    # Converti in grayscale se RGB
    if len(image.shape) == 3:
        print("    Conversione RGB -> Grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Converti a 8-bit se necessario
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
    else:
        image_8bit = image

    binary = cv2.adaptiveThreshold(
        image_8bit,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # Post-processing
    # Remove small objects
    min_size = 50
    binary_bool = binary > 0
    cleaned = morphology.remove_small_objects(binary_bool, min_size=min_size)

    # Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(cleaned.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilation
    dilated = cv2.dilate(closed, kernel, iterations=1)

    print(f"    Pixel mask: {np.sum(dilated > 0):,}")

    return dilated


def apply_morphological_closing(mask, kernel_size=15):
    """
    Applica morphological closing per chiudere gap.
    """
    print(f"  Kernel: {kernel_size}×{kernel_size} ellittico")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    n_pixels_added = np.sum(cv2.subtract(mask_closed, mask) > 0)
    print(f"    Pixel aggiunti: {n_pixels_added:,}")

    return mask_closed


def identify_closed_cycles(mask, min_area=1000, exclude_largest=True):
    """
    Identifica cicli chiusi e calcola statistiche per ogni fibra.
    """
    print(f"  Filtro area minima: {min_area} px²")

    # Binary fill holes
    mask_bool = mask > 0
    filled = ndi.binary_fill_holes(mask_bool).astype(np.uint8) * 255

    # Cicli = filled - mask
    cycles = cv2.subtract(filled, mask)

    # Find contours
    contours, _ = cv2.findContours(cycles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"    Contorni trovati: {len(contours)}")

    # Analizza ogni contorno
    fibers_data = []

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Calcola statistiche
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue

        centroid_x = M['m10'] / M['m00']
        centroid_y = M['m01'] / M['m00']
        perimeter = cv2.arcLength(contour, True)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Equivalent diameter
        equiv_diameter = np.sqrt(4 * area / np.pi)

        # Circularity
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0

        # Aspect ratio
        if h > 0:
            aspect_ratio = w / h
        else:
            aspect_ratio = 0

        fibers_data.append({
            'fiber_id': idx + 1,
            'area_px': area,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'perimeter_px': perimeter,
            'equiv_diameter_px': equiv_diameter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'bbox_x': x,
            'bbox_y': y,
            'bbox_width': w,
            'bbox_height': h,
        })

    # Ordina per area decrescente
    fibers_data.sort(key=lambda x: x['area_px'], reverse=True)

    # Escludi la più grande se richiesto (background)
    if exclude_largest and len(fibers_data) > 0:
        largest = fibers_data.pop(0)
        print(f"    Escluso ciclo più grande: {largest['area_px']:.0f} px² (background)")

    # Rinumera dopo esclusione
    for idx, fiber in enumerate(fibers_data):
        fiber['fiber_id'] = idx + 1

    print(f"    Fibre valide identificate: {len(fibers_data)}")

    return fibers_data, contours


def create_skeleton(mask):
    """
    Crea skeleton della maschera.
    """
    skeleton = morphology.skeletonize(mask > 0).astype(np.uint8) * 255
    return skeleton


def create_closing_visualization(mask_initial, mask_closed, output_dir):
    """
    Visualizza i pixel aggiunti dal morphological closing.
    """
    print("\nCreazione visualizzazione closing...")

    # Calcola pixel aggiunti
    pixels_added = cv2.subtract(mask_closed, mask_initial)

    # Crea immagine RGB
    # - Maschera originale: grigio
    # - Pixel aggiunti: blu chiaro (cyan)
    vis = np.zeros((mask_initial.shape[0], mask_initial.shape[1], 3), dtype=np.uint8)

    # Maschera originale in grigio
    vis[mask_initial > 0] = [128, 128, 128]

    # Pixel aggiunti in blu chiaro (cyan)
    vis[pixels_added > 0] = [255, 255, 0]  # Cyan in BGR

    cv2.imwrite(str(output_dir / 'closing_gaps_visualization.png'), vis)

    n_pixels_added = np.sum(pixels_added > 0)
    print(f"  Salvato: closing_gaps_visualization.png")
    print(f"  - Maschera originale: grigio")
    print(f"  - Pixel aggiunti ({n_pixels_added:,}): blu chiaro (cyan)")


def create_visualizations(fluorescence, mask_initial, mask_closed, skeleton, fibers_data, contours, output_dir, dot_radius=5):
    """
    Crea tutte le visualizzazioni.
    """
    print("\nCreazione visualizzazioni...")

    # Calcola pixel aggiunti (gap chiusi)
    pixels_added = cv2.subtract(mask_closed, mask_initial)

    # Prepara immagine base trasparente
    if len(fluorescence.shape) == 2:
        fluor_rgb = cv2.cvtColor(fluorescence, cv2.COLOR_GRAY2RGB)
    else:
        fluor_rgb = fluorescence.copy()

    if fluor_rgb.dtype == np.uint16:
        fluor_rgb = (fluor_rgb / 256).astype(np.uint8)

    # Prepara dati
    centroids = [(int(f['centroid_y']), int(f['centroid_x'])) for f in fibers_data]

    # === FASE 1: Segmentazione Iniziale ===
    print("  1. Segmentazione iniziale overlay...")
    seg_overlay = fluor_rgb.copy()
    seg_overlay = (seg_overlay * 0.5).astype(np.uint8)  # 50% trasparenza

    # Trova e disegna SOLO i contorni della segmentazione iniziale
    contours_initial, _ = cv2.findContours(mask_initial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_overlay, contours_initial, -1, (0, 255, 0), 2)  # Verde spessore 2

    cv2.imwrite(str(output_dir / 'step1_segmentation_overlay.png'), seg_overlay)
    print(f"     Salvato: step1_segmentation_overlay.png (contorni verdi = {len(contours_initial)} oggetti)")

    # === FASE 2: Morphological Closing ===
    print("  2. Morphological closing overlay...")
    closing_overlay = fluor_rgb.copy()
    closing_overlay = (closing_overlay * 0.5).astype(np.uint8)  # 50% trasparenza

    # Trova e disegna SOLO i contorni della maschera chiusa
    contours_closed, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(closing_overlay, contours_closed, -1, (0, 200, 255), 2)  # Arancione spessore 2

    # Gap aggiunti in cyan brillante
    closing_overlay[pixels_added > 0] = [255, 255, 0]  # Cyan per gap aggiunti

    cv2.imwrite(str(output_dir / 'step2_closing_overlay.png'), closing_overlay)
    print(f"     Salvato: step2_closing_overlay.png (arancione = contorni closing, cyan = gap)")

    # === FASE 3: Cicli Chiusi (solo quelli validi con centroidi) ===
    print("  3. Cicli chiusi overlay con contorni blu...")

    # Calcola i cicli chiusi (filled - mask)
    mask_bool = mask_closed > 0
    filled = ndi.binary_fill_holes(mask_bool).astype(np.uint8) * 255
    cycles = cv2.subtract(filled, mask_closed)

    # Trova contorni dei cicli
    cycles_contours, _ = cv2.findContours(cycles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crea maschera solo per i cicli validi (quelli in fibers_data)
    valid_cycles_mask = np.zeros_like(cycles)

    # Per ogni fibra in fibers_data, trova il contorno corrispondente
    for fiber in fibers_data:
        cx, cy = int(fiber['centroid_x']), int(fiber['centroid_y'])
        # Trova quale contorno contiene questo centroide
        for contour in cycles_contours:
            if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                cv2.drawContours(valid_cycles_mask, [contour], -1, 255, -1)
                break

    cycles_overlay = fluor_rgb.copy()
    cycles_overlay = (cycles_overlay * 0.5).astype(np.uint8)  # 50% trasparenza

    # Disegna SOLO i contorni dei cicli validi in blu brillante
    for fiber in fibers_data:
        cx, cy = int(fiber['centroid_x']), int(fiber['centroid_y'])
        for contour in cycles_contours:
            if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                cv2.drawContours(cycles_overlay, [contour], -1, (255, 0, 0), 2)  # Blu spessore 2
                break

    # Disegna pallini rossi per i centroidi
    for cy, cx in centroids:
        cv2.circle(cycles_overlay, (cx, cy), dot_radius, (0, 0, 255), -1)

    cv2.imwrite(str(output_dir / 'step3_cycles_overlay.png'), cycles_overlay)
    print(f"     Salvato: step3_cycles_overlay.png ({len(centroids)} contorni blu + centroidi rossi)")

    # === FASE 4: Skeletonizzazione ===
    print("  4. Skeleton overlay con profili blu...")
    skeleton_overlay = fluor_rgb.copy()
    skeleton_overlay = (skeleton_overlay * 0.5).astype(np.uint8)  # 50% trasparenza

    # Skeleton in blu brillante
    skeleton_overlay[skeleton > 0] = [255, 0, 0]  # Blu

    # Disegna pallini rossi per i centroidi
    for cy, cx in centroids:
        cv2.circle(skeleton_overlay, (cx, cy), dot_radius, (0, 0, 255), -1)

    cv2.imwrite(str(output_dir / 'step4_skeleton_overlay.png'), skeleton_overlay)
    print(f"     Salvato: step4_skeleton_overlay.png (blu = skeleton, rosso = centroidi)")

    # === VISUALIZZAZIONI LEGACY (compatibilità) ===
    print("  5. Laminina con centroidi e gap...")
    laminina_overlay = fluor_rgb.copy()

    # Sovrapponi gap in blu chiaro (cyan)
    laminina_overlay[pixels_added > 0] = [255, 255, 0]  # Cyan (blu chiaro)

    # Disegna pallini rossi sopra
    for cy, cx in centroids:
        cv2.circle(laminina_overlay, (cx, cy), dot_radius, (0, 0, 255), -1)

    cv2.imwrite(str(output_dir / 'laminina_with_centroids.png'), laminina_overlay)
    print(f"     Salvato: laminina_with_centroids.png ({len(centroids)} fibre + gap cyan)")

    print("  6. Circuiti chiusi con centroidi e gap...")
    # Crea immagine RGB con solo i cicli chiusi validi (bianco su nero)
    cycles_rgb = np.zeros_like(fluor_rgb)
    cycles_rgb[valid_cycles_mask > 0] = [255, 255, 255]  # Bianco per cicli

    # Sovrapponi gap in blu chiaro (cyan)
    cycles_rgb[pixels_added > 0] = [255, 255, 0]  # Cyan (blu chiaro)

    # Disegna pallini rossi sopra per i centroidi
    for cy, cx in centroids:
        cv2.circle(cycles_rgb, (cx, cy), dot_radius, (0, 0, 255), -1)

    cv2.imwrite(str(output_dir / 'skeleton_with_centroids.png'), cycles_rgb)
    print(f"     Salvato: skeleton_with_centroids.png ({len(centroids)} circuiti + gap cyan)")

    # 7. Istogramma aree
    print("  7. Istogramma distribuzione aree...")
    areas = [f['area_px'] for f in fibers_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(areas, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Area (pixel²)', fontsize=12)
    ax.set_ylabel('Numero di fibre', fontsize=12)
    ax.set_title(f'Distribuzione Aree Fibre Muscolari (n={len(areas)})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Statistiche sul grafico
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    ax.axvline(mean_area, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_area:.0f} px²')
    ax.axvline(median_area, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_area:.0f} px²')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'area_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"     Salvato: area_distribution.png")


def save_statistics(fibers_data, mask, output_dir, image_shape, pixel_size_um=None):
    """
    Salva statistiche in formato CSV.

    Args:
        pixel_size_um: Dimensione pixel in µm (es. 0.41026 per 2.4375 px/µm)
    """
    print("\nSalvataggio statistiche...")

    # 1. CSV con dati per ogni fibra
    df_fibers = pd.DataFrame(fibers_data)

    # Ordina per area decrescente
    df_fibers = df_fibers.sort_values('area_px', ascending=False).reset_index(drop=True)
    df_fibers['fiber_id'] = df_fibers.index + 1

    # Aggiungi colonne calibrate se disponibile
    if pixel_size_um is not None:
        df_fibers['area_um2'] = df_fibers['area_px'] * (pixel_size_um ** 2)
        df_fibers['perimeter_um'] = df_fibers['perimeter_px'] * pixel_size_um
        df_fibers['equiv_diameter_um'] = df_fibers['equiv_diameter_px'] * pixel_size_um
        df_fibers['bbox_width_um'] = df_fibers['bbox_width'] * pixel_size_um
        df_fibers['bbox_height_um'] = df_fibers['bbox_height'] * pixel_size_um
        print(f"  Calibrazione applicata: 1 pixel = {pixel_size_um:.6f} µm")

    csv_path = output_dir / 'fibers_statistics.csv'
    df_fibers.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"  Salvato: fibers_statistics.csv ({len(df_fibers)} righe)")

    # 2. CSV con statistiche sommarie
    areas = df_fibers['area_px'].values

    summary = {
        'metric': [
            'n_fibers',
            'total_area_px',
            'mean_area_px',
            'median_area_px',
            'std_area_px',
            'min_area_px',
            'max_area_px',
            'q25_area_px',
            'q75_area_px',
            'image_width_px',
            'image_height_px',
            'image_total_px',
            'mask_coverage_percent'
        ],
        'value': [
            len(df_fibers),
            np.sum(areas),
            np.mean(areas),
            np.median(areas),
            np.std(areas),
            np.min(areas),
            np.max(areas),
            np.percentile(areas, 25),
            np.percentile(areas, 75),
            image_shape[1],
            image_shape[0],
            image_shape[0] * image_shape[1],
            (np.sum(mask > 0) / (image_shape[0] * image_shape[1]) * 100)
        ]
    }

    # Aggiungi statistiche calibrate
    if pixel_size_um is not None:
        areas_um2 = df_fibers['area_um2'].values
        summary['metric'].extend([
            'pixel_size_um',
            'total_area_um2',
            'mean_area_um2',
            'median_area_um2',
            'std_area_um2',
            'min_area_um2',
            'max_area_um2',
            'q25_area_um2',
            'q75_area_um2'
        ])
        summary['value'].extend([
            pixel_size_um,
            np.sum(areas_um2),
            np.mean(areas_um2),
            np.median(areas_um2),
            np.std(areas_um2),
            np.min(areas_um2),
            np.max(areas_um2),
            np.percentile(areas_um2, 25),
            np.percentile(areas_um2, 75)
        ])

    df_summary = pd.DataFrame(summary)
    summary_path = output_dir / 'summary_statistics.csv'
    df_summary.to_csv(summary_path, index=False, float_format='%.2f')
    print(f"  Salvato: summary_statistics.csv")

    # 3. JSON con metadati completi
    metadata = {
        'n_fibers': int(len(df_fibers)),
        'image_dimensions': {'width': int(image_shape[1]), 'height': int(image_shape[0])},
        'area_statistics_px': {
            'mean': float(np.mean(areas)),
            'median': float(np.median(areas)),
            'std': float(np.std(areas)),
            'min': float(np.min(areas)),
            'max': float(np.max(areas)),
            'q25': float(np.percentile(areas, 25)),
            'q75': float(np.percentile(areas, 75))
        },
        'mask_coverage_percent': float(np.sum(mask > 0) / (image_shape[0] * image_shape[1]) * 100)
    }

    # Aggiungi calibrazione al metadata
    if pixel_size_um is not None:
        areas_um2 = df_fibers['area_um2'].values
        metadata['calibration'] = {
            'pixel_size_um': pixel_size_um,
            'pixels_per_um': 1.0 / pixel_size_um
        }
        metadata['area_statistics_um2'] = {
            'mean': float(np.mean(areas_um2)),
            'median': float(np.median(areas_um2)),
            'std': float(np.std(areas_um2)),
            'min': float(np.min(areas_um2)),
            'max': float(np.max(areas_um2)),
            'q25': float(np.percentile(areas_um2, 25)),
            'q75': float(np.percentile(areas_um2, 75))
        }

    json_path = output_dir / 'metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Salvato: metadata.json")

    return df_fibers, df_summary


def print_summary(fibers_data, image_shape, pixel_size_um=None):
    """
    Stampa riepilogo a schermo.
    """
    areas = [f['area_px'] for f in fibers_data]

    print("\n" + "="*80)
    print("RISULTATI ANALISI")
    print("="*80)
    print(f"Fibre muscolari identificate:  {len(fibers_data):,}")
    print(f"\nArea (pixel):")
    print(f"  Media:     {np.mean(areas):,.0f} px²")
    print(f"  Mediana:   {np.median(areas):,.0f} px²")
    print(f"  Dev Std:   {np.std(areas):,.0f} px²")
    print(f"  Range:     {np.min(areas):,.0f} - {np.max(areas):,.0f} px²")
    print(f"  Q1, Q3:    {np.percentile(areas, 25):,.0f}, {np.percentile(areas, 75):,.0f} px²")

    if pixel_size_um is not None:
        areas_um2 = [a * (pixel_size_um ** 2) for a in areas]
        print(f"\nArea (µm²):")
        print(f"  Media:     {np.mean(areas_um2):,.2f} µm²")
        print(f"  Mediana:   {np.median(areas_um2):,.2f} µm²")
        print(f"  Dev Std:   {np.std(areas_um2):,.2f} µm²")
        print(f"  Range:     {np.min(areas_um2):,.2f} - {np.max(areas_um2):,.2f} µm²")
        print(f"  Q1, Q3:    {np.percentile(areas_um2, 25):,.2f}, {np.percentile(areas_um2, 75):,.2f} µm²")
        print(f"\nCalibrazione: 1 pixel = {pixel_size_um:.6f} µm ({1/pixel_size_um:.4f} px/µm)")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline completa analisi fibre muscolari da fluorescenza laminina',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Analisi senza calibrazione (solo pixel)
  python scripts/analyze_laminina_complete.py \\
      --input data/laminina_originale.png \\
      --output output_final \\
      --kernel-size 15 \\
      --min-fiber-area 1000 \\
      --dot-radius 5

  # Analisi con calibrazione (pixel + µm²)
  # Per 2.4375 px/µm -> pixel-size = 1/2.4375 = 0.41026 µm
  python scripts/analyze_laminina_complete.py \\
      --input data/laminina_originale.png \\
      --output output_final \\
      --kernel-size 15 \\
      --min-fiber-area 1000 \\
      --dot-radius 5 \\
      --pixel-size 0.41026
        """
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Immagine fluorescenza laminina (PNG, TIFF, ecc.)')
    parser.add_argument('--output', type=str, default='output_final',
                       help='Directory di output (default: output_final)')
    parser.add_argument('--kernel-size', type=int, default=15,
                       help='Dimensione kernel closing (default: 15)')
    parser.add_argument('--min-fiber-area', type=int, default=1000,
                       help='Area minima fibra in pixel² (default: 1000)')
    parser.add_argument('--dot-radius', type=int, default=5,
                       help='Raggio pallini centroidi (default: 5)')
    parser.add_argument('--block-size', type=int, default=101,
                       help='Block size adaptive threshold (default: 101)')
    parser.add_argument('--threshold-C', type=int, default=2,
                       help='Costante C adaptive threshold (default: 2)')
    parser.add_argument('--pixel-size', type=float, default=None,
                       help='Dimensione pixel in µm (es. 0.41026 per calibrazione 2.4375 px/µm). Se omesso, risultati solo in pixel.')

    args = parser.parse_args()

    print("="*80)
    print("ANALISI COMPLETA FIBRE MUSCOLARI - FLUORESCENZA LAMININA")
    print("="*80)

    # Crea output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # STEP 1: Caricamento
    print(f"\n{'STEP 1: CARICAMENTO':-^80}")
    print(f"Input: {args.input}")

    fluorescence = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if fluorescence is None:
        raise ValueError(f"Impossibile caricare: {args.input}")

    print(f"  Dimensioni: {fluorescence.shape}")
    print(f"  Dtype: {fluorescence.dtype}")
    print(f"  Range valori: {fluorescence.min()} - {fluorescence.max()}")

    # STEP 2: Segmentazione
    print(f"\n{'STEP 2: SEGMENTAZIONE':-^80}")
    mask_initial = segment_adaptive_threshold(
        fluorescence,
        block_size=args.block_size,
        C=args.threshold_C
    )

    # STEP 3: Morphological Closing
    print(f"\n{'STEP 3: MORPHOLOGICAL CLOSING':-^80}")
    mask_closed = apply_morphological_closing(mask_initial, kernel_size=args.kernel_size)

    # STEP 4: Identificazione cicli
    print(f"\n{'STEP 4: IDENTIFICAZIONE CICLI':-^80}")
    fibers_data, contours = identify_closed_cycles(
        mask_closed,
        min_area=args.min_fiber_area,
        exclude_largest=True
    )

    if len(fibers_data) == 0:
        print("\n⚠️  ATTENZIONE: Nessuna fibra identificata!")
        print("   Prova a ridurre --min-fiber-area o modificare parametri segmentazione.")
        return

    # STEP 5: Skeletonizzazione
    print(f"\n{'STEP 5: SKELETONIZZAZIONE':-^80}")
    print("  Creazione skeleton...")
    skeleton = create_skeleton(mask_closed)

    # STEP 6: Visualizzazioni
    print(f"\n{'STEP 6: VISUALIZZAZIONI':-^80}")

    # Visualizzazione closing gaps
    create_closing_visualization(mask_initial, mask_closed, output_dir)

    create_visualizations(
        fluorescence, mask_initial, mask_closed, skeleton, fibers_data, contours,
        output_dir, dot_radius=args.dot_radius
    )

    # STEP 7: Statistiche
    print(f"\n{'STEP 7: ESPORTAZIONE STATISTICHE':-^80}")
    df_fibers, df_summary = save_statistics(
        fibers_data, mask_closed, output_dir, fluorescence.shape, pixel_size_um=args.pixel_size
    )

    # STEP 8: Riepilogo
    print_summary(fibers_data, fluorescence.shape, pixel_size_um=args.pixel_size)

    print(f"\n{'COMPLETATO':-^80}")
    print(f"Tutti i file sono stati salvati in: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()