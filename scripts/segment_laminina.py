#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per segmentazione automatica di immagini di fluorescenza della laminina.

STEP 1: Creare maschera automatica da immagine fluorescenza originale
        (simile a quella ottenuta manualmente con Photoshop/ImageJ)

STEP 2: Chiudere cicli usando multi-threshold e informazioni di intensità

Uso:
    python scripts/segment_laminina.py --input data/laminina_originale.png --reference data/laminina_maschera_reference.png --output output_segmentation
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters, morphology, exposure
from skimage.morphology import remove_small_objects, disk


def load_and_analyze_reference(reference_path):
    """
    Carica e analizza la maschera di riferimento per capire le caratteristiche.

    Returns:
        Dict con statistiche della maschera di riferimento
    """
    print(f"Caricamento maschera di riferimento: {reference_path}")
    ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)

    if ref is None:
        raise ValueError(f"Impossibile caricare: {reference_path}")

    print(f"  Dimensioni: {ref.shape}")
    print(f"  Range valori: {ref.min()} - {ref.max()}")
    print(f"  Tipo: {ref.dtype}")

    # Analizza la maschera
    ref_binary = ref > 127
    n_white_pixels = np.sum(ref_binary)
    n_total_pixels = ref_binary.size
    white_percentage = (n_white_pixels / n_total_pixels) * 100

    print(f"  Pixel bianchi: {n_white_pixels:,} ({white_percentage:.2f}%)")

    # Trova contorni
    contours, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Numero di regioni connesse: {len(contours)}")

    # Calcola spessore medio dei contorni (stima)
    labeled = morphology.label(ref_binary)
    n_regions = labeled.max()
    print(f"  Regioni connesse (label): {n_regions}")

    stats = {
        'shape': ref.shape,
        'white_percentage': white_percentage,
        'n_contours': len(contours),
        'n_regions': n_regions
    }

    return ref, stats


def segment_adaptive_threshold(image, block_size=101, C=2):
    """
    Segmentazione usando adaptive thresholding.
    Buono per immagini con intensità variabile.
    """
    print(f"Segmentazione con adaptive threshold (block_size={block_size}, C={C})...")

    # Normalizza immagine a 8-bit se necessario
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
    else:
        image_8bit = image

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        image_8bit,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    return binary


def segment_clahe_otsu(image, clip_limit=2.0, tile_size=8):
    """
    Segmentazione usando CLAHE (equalizzazione contrasto) + Otsu threshold.
    """
    print(f"Segmentazione con CLAHE + Otsu (clip_limit={clip_limit}, tile_size={tile_size})...")

    # Normalizza a 8-bit se necessario
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
    else:
        image_8bit = image

    # CLAHE per equalizzare contrasto
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    equalized = clahe.apply(image_8bit)

    # Otsu threshold
    _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def segment_morphological_gradient(image, threshold_percentile=50):
    """
    Segmentazione usando gradiente morfologico.
    Identifica i bordi/contorni delle fibre.
    """
    print(f"Segmentazione con gradiente morfologico (threshold_percentile={threshold_percentile})...")

    # Normalizza a 8-bit
    if image.dtype == np.uint16:
        image_8bit = (image / 256).astype(np.uint8)
    else:
        image_8bit = image

    # Gradiente morfologico (rileva bordi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(image_8bit, cv2.MORPH_GRADIENT, kernel)

    # Threshold sul gradiente
    threshold_value = np.percentile(gradient[gradient > 0], threshold_percentile)
    binary = (gradient > threshold_value).astype(np.uint8) * 255

    return binary


def postprocess_mask(binary, min_object_size=500, close_size=3, dilate_size=2):
    """
    Post-processing della maschera binaria.

    Args:
        binary: Maschera binaria (0 o 255)
        min_object_size: Rimuovi oggetti più piccoli di questo (pixel)
        close_size: Dimensione kernel per closing morfologico
        dilate_size: Dimensione kernel per dilatazione (ispessimento)
    """
    print(f"Post-processing maschera...")
    print(f"  Rimozione oggetti piccoli (< {min_object_size} px)...")

    # Converti a booleano
    binary_bool = binary > 0

    # Rimuovi oggetti piccoli
    cleaned = remove_small_objects(binary_bool, min_size=min_object_size)

    # Closing morfologico (chiude gap piccoli)
    if close_size > 0:
        print(f"  Closing morfologico (kernel={close_size})...")
        kernel = disk(close_size)
        cleaned = morphology.binary_closing(cleaned, kernel)

    # Dilatazione (ispessisce i contorni)
    if dilate_size > 0:
        print(f"  Dilatazione (kernel={dilate_size})...")
        kernel = disk(dilate_size)
        cleaned = morphology.binary_dilation(cleaned, kernel)

    # Converti a uint8
    result = (cleaned * 255).astype(np.uint8)

    return result


def compare_with_reference(predicted, reference):
    """
    Confronta la maschera predetta con la maschera di riferimento.
    Calcola metriche di similarità.
    """
    print("Confronto con maschera di riferimento...")

    pred_bool = predicted > 127
    ref_bool = reference > 127

    # Intersection over Union (IoU) / Jaccard
    intersection = np.logical_and(pred_bool, ref_bool).sum()
    union = np.logical_or(pred_bool, ref_bool).sum()
    iou = intersection / union if union > 0 else 0

    # Dice coefficient
    dice = (2 * intersection) / (pred_bool.sum() + ref_bool.sum()) if (pred_bool.sum() + ref_bool.sum()) > 0 else 0

    # Precision e Recall
    true_positive = intersection
    false_positive = np.logical_and(pred_bool, np.logical_not(ref_bool)).sum()
    false_negative = np.logical_and(np.logical_not(pred_bool), ref_bool).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  IoU (Jaccard): {iou:.4f}")
    print(f"  Dice: {dice:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def visualize_comparison(original, reference, predicted, metrics, output_path):
    """
    Crea visualizzazione comparativa tra riferimento e predizione.
    """
    print(f"Creazione visualizzazione comparativa...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Riga 1: Immagini singole
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Immagine Originale (Fluorescenza)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(reference, cmap='gray')
    axes[0, 1].set_title('Maschera Riferimento (Photoshop/ImageJ)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(predicted, cmap='gray')
    axes[0, 2].set_title('Maschera Predetta (Automatica)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Riga 2: Confronti
    # Overlay
    overlay = np.zeros((*original.shape, 3), dtype=np.uint8)
    overlay[reference > 127] = [0, 255, 0]  # Verde: riferimento
    overlay[predicted > 127] = [255, 0, 0]  # Rosso: predetto
    overlay[np.logical_and(reference > 127, predicted > 127)] = [255, 255, 0]  # Giallo: overlap

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (Verde=Rif, Rosso=Pred, Giallo=Overlap)', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')

    # Differenza
    diff = np.zeros_like(original)
    diff[np.logical_and(reference > 127, predicted <= 127)] = 128  # Falsi negativi
    diff[np.logical_and(reference <= 127, predicted > 127)] = 255  # Falsi positivi

    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Differenze (Chiaro=FP, Scuro=FN)', fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')

    # Metriche
    axes[1, 2].axis('off')
    metrics_text = f"""METRICHE DI SIMILARITÀ

IoU (Jaccard): {metrics['iou']:.4f}
Dice: {metrics['dice']:.4f}

Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1-score: {metrics['f1']:.4f}

Interpretazione:
- IoU/Dice: quanto sono simili (1=perfetto)
- Precision: % predetti corretti
- Recall: % riferimento catturato
"""
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Salvato: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Segmentazione automatica immagini fluorescenza laminina'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Immagine originale fluorescenza laminina'
    )
    parser.add_argument(
        '--reference',
        type=str,
        required=True,
        help='Maschera di riferimento (fatta con Photoshop/ImageJ)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_segmentation',
        help='Directory di output (default: output_segmentation)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='all',
        choices=['adaptive', 'clahe', 'gradient', 'all'],
        help='Metodo di segmentazione da usare (default: all, prova tutti)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SEGMENTAZIONE AUTOMATICA LAMININA")
    print("=" * 80)

    # Crea directory output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carica immagini
    print(f"\n1. CARICAMENTO IMMAGINI")
    print("-" * 80)

    print(f"Caricamento immagine originale: {args.input}")
    original = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Impossibile caricare: {args.input}")
    print(f"  Dimensioni: {original.shape}")
    print(f"  Range: {original.min()} - {original.max()}, dtype: {original.dtype}")

    reference, ref_stats = load_and_analyze_reference(args.reference)

    # STEP 1: Segmentazione con vari metodi
    print(f"\n2. SEGMENTAZIONE AUTOMATICA")
    print("-" * 80)

    results = {}

    if args.method in ['adaptive', 'all']:
        mask_adaptive = segment_adaptive_threshold(original, block_size=101, C=2)
        mask_adaptive = postprocess_mask(mask_adaptive, min_object_size=500, close_size=3, dilate_size=2)
        results['adaptive'] = mask_adaptive

        # Salva
        cv2.imwrite(str(output_dir / 'mask_adaptive.png'), mask_adaptive)
        print(f"  Salvato: {output_dir / 'mask_adaptive.png'}")

    if args.method in ['clahe', 'all']:
        mask_clahe = segment_clahe_otsu(original, clip_limit=2.0, tile_size=8)
        mask_clahe = postprocess_mask(mask_clahe, min_object_size=500, close_size=3, dilate_size=2)
        results['clahe'] = mask_clahe

        # Salva
        cv2.imwrite(str(output_dir / 'mask_clahe.png'), mask_clahe)
        print(f"  Salvato: {output_dir / 'mask_clahe.png'}")

    if args.method in ['gradient', 'all']:
        mask_gradient = segment_morphological_gradient(original, threshold_percentile=50)
        mask_gradient = postprocess_mask(mask_gradient, min_object_size=500, close_size=3, dilate_size=2)
        results['gradient'] = mask_gradient

        # Salva
        cv2.imwrite(str(output_dir / 'mask_gradient.png'), mask_gradient)
        print(f"  Salvato: {output_dir / 'mask_gradient.png'}")

    # STEP 2: Confronto con riferimento
    print(f"\n3. CONFRONTO CON RIFERIMENTO")
    print("-" * 80)

    best_method = None
    best_score = 0

    for method_name, mask in results.items():
        print(f"\nMetodo: {method_name.upper()}")
        metrics = compare_with_reference(mask, reference)

        # Visualizzazione
        vis_path = output_dir / f'comparison_{method_name}.png'
        visualize_comparison(original, reference, mask, metrics, vis_path)

        # Traccia il migliore (usando F1-score)
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_method = method_name

    print(f"\n" + "=" * 80)
    print(f"MIGLIORE METODO: {best_method.upper()} (F1-score: {best_score:.4f})")
    print("=" * 80)

    print(f"\nFile salvati in: {output_dir}")
    print("- mask_*.png: Maschere generate")
    print("- comparison_*.png: Visualizzazioni comparative")


if __name__ == '__main__':
    main()
