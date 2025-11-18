#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline completa per analisi fibre muscolari da immagine fluorescenza laminina.

STEP 1: Segmentazione automatica (adaptive threshold)
STEP 2: Morphological closing (default K15)
STEP 3: Identificazione cicli chiusi
STEP 4: Calcolo statistiche ed esportazione

Output:
  === VISUALIZZAZIONI ===
  - skeleton_before_closing.png               # Contorni CICLI CHIUSI PRIMA del closing (blu) su fluorescenza 100%
                                               # Solo anelli di laminina, NO rametti interni
                                               # NO centroidi
  - skeleton_after_closing_with_gaps.png      # Contorni CICLI CHIUSI DOPO del closing (blu) + gap aggiunti (cyan) + centroidi colorati
                                               # su fluorescenza 100%
                                               # Solo anelli di laminina chiusi, NO rametti interni
                                               # Cicli NUOVI: riempiti arancione/salmone chiaro (35% trasparenza)
                                               # Centroidi al centro degli anelli
                                               # Centroidi ROSSI: cicli già esistenti prima del closing
                                               # Centroidi GIALLI: cicli NUOVI chiusi dal morphological closing

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

    NOTA IMPORTANTE - Chiusura Geometrica vs Anatomica:
    Il closing riempie i gap con forme ellittiche. Per gap molto grandi (>15 px),
    la forma chiusa può non corrispondere alla forma biologica originale (il closing "inventa"
    una chiusura geometrica, non biologica). I cicli chiusi perdono la loro circolarità
    naturale e diventano più geometrici/ellittici.

    RACCOMANDAZIONE: Se i cicli chiusi non mantengono la forma anatomica originale,
    ridurre --kernel-size a valori più piccoli (es. 7, 9, 11) per maggiore fedeltà
    alla forma biologica, accettando che alcuni gap rimangano aperti.

    ESEMPIO:
      --kernel-size 15  → chiude gap grandi, ma forma geometrica
      --kernel-size 7   → forma più anatomica, ma gap grandi rimangono aperti
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
    valid_contours = []  # Salva solo i contorni validi

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
        valid_contours.append(contour)

    # Ordina per area decrescente (e riordina contorni di conseguenza)
    sorted_indices = sorted(range(len(fibers_data)), key=lambda i: fibers_data[i]['area_px'], reverse=True)
    fibers_data = [fibers_data[i] for i in sorted_indices]
    valid_contours = [valid_contours[i] for i in sorted_indices]

    # Escludi la più grande se richiesto (background)
    if exclude_largest and len(fibers_data) > 0:
        largest = fibers_data.pop(0)
        valid_contours.pop(0)  # Rimuovi anche il contorno corrispondente
        print(f"    Escluso ciclo più grande: {largest['area_px']:.0f} px² (background)")

    # Rinumera dopo esclusione
    for idx, fiber in enumerate(fibers_data):
        fiber['fiber_id'] = idx + 1

    print(f"    Fibre valide identificate: {len(fibers_data)}")

    return fibers_data, valid_contours


def create_skeleton(mask):
    """
    Crea skeleton della maschera.
    """
    skeleton = morphology.skeletonize(mask > 0).astype(np.uint8) * 255
    return skeleton




def create_visualizations(fluorescence, mask_initial, mask_closed, skeleton, fibers_data, contours, output_dir, dot_radius=5):
    """
    Crea solo 2 visualizzazioni essenziali:
    1. Contorni CICLI CHIUSI PRIMA del closing (senza centroidi) - solo anelli, NO rametti
    2. Contorni CICLI CHIUSI DOPO del closing + gap cyan + centroidi colorati - anelli chiusi con centroidi al centro
    """
    print("\nCreazione visualizzazioni...")

    # Prepara immagine base
    if len(fluorescence.shape) == 2:
        fluor_rgb = cv2.cvtColor(fluorescence, cv2.COLOR_GRAY2RGB)
    else:
        fluor_rgb = fluorescence.copy()

    if fluor_rgb.dtype == np.uint16:
        fluor_rgb = (fluor_rgb / 256).astype(np.uint8)

    # Prepara centroidi
    centroids = [(int(f['centroid_y']), int(f['centroid_x'])) for f in fibers_data]

    # ========================================================================
    # IMMAGINE 1: Contorni CICLI CHIUSI PRIMA del morphological closing (SENZA centroidi)
    # ========================================================================
    print("  1. Contorni cicli chiusi PRIMA del closing (senza centroidi)...")

    # Calcola cicli PRIMA del closing (per trovare gli anelli di laminina)
    mask_initial_bool = mask_initial > 0
    filled_before = ndi.binary_fill_holes(mask_initial_bool).astype(np.uint8) * 255
    cycles_before = cv2.subtract(filled_before, mask_initial)

    # Trova contorni dei cicli chiusi (gli anelli di laminina)
    contours_cycles_before, _ = cv2.findContours(cycles_before, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"     Trovati {len(contours_cycles_before)} cicli chiusi nella laminina iniziale")

    # Crea immagine: fluorescenza + contorni cicli in blu
    before_overlay = fluor_rgb.copy()  # 100% opacità

    # Disegna solo i contorni dei cicli chiusi in blu (spessore 2px per visibilità)
    cv2.drawContours(before_overlay, contours_cycles_before, -1, (255, 0, 0), 2)  # Blu

    cv2.imwrite(str(output_dir / 'skeleton_before_closing.png'), before_overlay)
    print(f"     Salvato: skeleton_before_closing.png")
    print(f"        - Fluorescenza 100% opacità")
    print(f"        - Contorni cicli chiusi blu: {len(contours_cycles_before)} anelli")
    print(f"        - NO centroidi")
    print(f"        - Solo anelli di laminina, NO rametti interni")

    # ========================================================================
    # IMMAGINE 2: Contorni CICLI CHIUSI DOPO del closing + gap cyan + centroidi colorati
    # ========================================================================
    print("  2. Contorni cicli chiusi DOPO del closing + gap + centroidi...")

    # Calcola cicli DOPO il closing (per trovare gli anelli di laminina chiusi)
    mask_closed_bool = mask_closed > 0
    filled_after = ndi.binary_fill_holes(mask_closed_bool).astype(np.uint8) * 255
    cycles_after = cv2.subtract(filled_after, mask_closed)

    # Trova contorni dei cicli chiusi dopo closing
    contours_cycles_after, _ = cv2.findContours(cycles_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"     Trovati {len(contours_cycles_after)} cicli chiusi dopo closing")

    # Calcola pixel aggiunti dal closing (gap chiusi)
    pixels_added = cv2.subtract(mask_closed, mask_initial)
    n_gap = np.sum(pixels_added > 0)

    # Identifica cicli PRIMA del closing per distinguere nuovi vs esistenti
    # Riusa contours_cycles_before già calcolato
    print("     Identificazione cicli nuovi vs esistenti...")
    contours_before = contours_cycles_before  # Riusa i contorni già calcolati

    # Per ogni fibra (dopo closing), verifica se il centroide cade in un ciclo che esisteva prima
    new_fibers = []  # Cicli nuovi (chiusi dal closing)
    existing_fibers = []  # Cicli già esistenti prima

    for fiber in fibers_data:
        cx, cy = fiber['centroid_x'], fiber['centroid_y']
        point = (int(cx), int(cy))

        # Verifica se il centroide cade dentro un ciclo che esisteva prima
        is_existing = False
        for contour in contours_before:
            result = cv2.pointPolygonTest(contour, point, False)
            if result >= 0:  # Punto dentro o sul contorno
                is_existing = True
                break

        if is_existing:
            existing_fibers.append(fiber)
        else:
            new_fibers.append(fiber)

    print(f"        Cicli già esistenti prima: {len(existing_fibers)}")
    print(f"        Cicli NUOVI (chiusi dal closing): {len(new_fibers)}")

    # Crea immagine: fluorescenza + contorni cicli chiusi blu + gap cyan + centroidi colorati
    after_overlay = fluor_rgb.copy()  # 100% opacità

    # RIEMPI i cicli NUOVI con colore semi-trasparente per evidenziarli
    if len(new_fibers) > 0:
        print(f"     Riempimento cicli nuovi con colore semi-trasparente...")

        # Crea maschera per cicli nuovi
        new_cycles_mask = np.zeros_like(cycles_after)

        # Per ogni fibra nuova, trova il contorno corrispondente e riempilo
        for fiber in new_fibers:
            cx, cy = int(fiber['centroid_x']), int(fiber['centroid_y'])
            point = (cx, cy)

            # Trova quale contorno contiene questo centroide
            for contour in contours_cycles_after:
                result = cv2.pointPolygonTest(contour, point, False)
                if result >= 0:  # Centroide dentro questo contorno
                    cv2.drawContours(new_cycles_mask, [contour], -1, 255, -1)  # Riempi
                    break

        # Crea overlay semi-trasparente arancione/salmone chiaro
        overlay_color = np.zeros_like(after_overlay)
        overlay_color[new_cycles_mask > 0] = [100, 200, 255]  # Arancione/salmone chiaro in BGR

        # Mischia con trasparenza 35%
        alpha = 0.35
        after_overlay[new_cycles_mask > 0] = (
            alpha * overlay_color[new_cycles_mask > 0] +
            (1 - alpha) * after_overlay[new_cycles_mask > 0]
        ).astype(np.uint8)

    # Disegna contorni cicli chiusi dopo closing in blu
    cv2.drawContours(after_overlay, contours_cycles_after, -1, (255, 0, 0), 2)  # Blu

    # Sovrascrivi pixel aggiunti (gap chiusi) in cyan
    after_overlay[pixels_added > 0] = [255, 255, 0]  # Cyan

    # Disegna centroidi con colori diversi
    # ROSSO: cicli già esistenti prima del closing
    for fiber in existing_fibers:
        cy, cx = int(fiber['centroid_y']), int(fiber['centroid_x'])
        cv2.circle(after_overlay, (cx, cy), dot_radius, (0, 0, 255), -1)  # Rosso

    # GIALLO: cicli NUOVI chiusi dal morphological closing
    for fiber in new_fibers:
        cy, cx = int(fiber['centroid_y']), int(fiber['centroid_x'])
        cv2.circle(after_overlay, (cx, cy), dot_radius, (0, 255, 255), -1)  # Giallo (contrasta con verde laminina)

    cv2.imwrite(str(output_dir / 'skeleton_after_closing_with_gaps.png'), after_overlay)
    print(f"     Salvato: skeleton_after_closing_with_gaps.png")
    print(f"        - Fluorescenza 100% opacità")
    print(f"        - Contorni cicli chiusi blu: {len(contours_cycles_after)} anelli")
    print(f"        - Gap aggiunti cyan: {n_gap:,} pixel")
    print(f"        - Cicli NUOVI riempiti arancione/salmone chiaro (35% trasparenza)")
    print(f"        - Centroidi ROSSI: {len(existing_fibers)} (cicli già esistenti)")
    print(f"        - Centroidi GIALLI: {len(new_fibers)} (cicli nuovi chiusi dal closing)")
    print(f"        - Solo anelli di laminina chiusi, NO rametti interni")


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