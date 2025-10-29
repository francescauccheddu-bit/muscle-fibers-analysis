#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per analizzare contorni di fibre muscolari in sezione.

1. Pulisce il rumore dalla maschera binaria
2. Crea lo scheletro (linee di 1 pixel) dei contorni
3. Identifica i cicli chiusi (percorsi che tornano su se stessi)
4. Riempie le regioni dentro i cicli chiusi con verde
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage as ndi


def clean_and_skeletonize(binary_mask, min_area=500, border_exclusion=0):
    """
    Pulisce la maschera dal rumore e crea lo scheletro dei contorni.

    Args:
        binary_mask: Maschera binaria con contorni
        min_area: Area minima per mantenere un oggetto
        border_exclusion: Larghezza del bordo da escludere

    Returns:
        skeleton: Scheletro dei contorni (linee di 1 pixel)
        cleaned: Maschera pulita prima della scheletonizzazione
    """
    print("Pulizia maschera...")

    # Converti in booleano
    mask_bool = binary_mask > 0

    # Rimuovi piccoli oggetti (rumore)
    cleaned = remove_small_objects(mask_bool, min_size=min_area)

    # Escludi bordi se richiesto
    if border_exclusion > 0:
        print(f"Esclusione bordo di {border_exclusion} pixel...")
        h, w = cleaned.shape
        cleaned[:border_exclusion, :] = False
        cleaned[-border_exclusion:, :] = False
        cleaned[:, :border_exclusion] = False
        cleaned[:, -border_exclusion:] = False

    # Conta oggetti rimasti
    labeled = measure.label(cleaned)
    n_objects = labeled.max()
    print(f"Oggetti dopo pulizia: {n_objects}")

    # Scheletonizza
    print("Creazione scheletro...")
    skeleton = skeletonize(cleaned)

    # Converti in uint8 per visualizzazione
    cleaned_uint8 = (cleaned * 255).astype(np.uint8)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    return skeleton_uint8, cleaned_uint8


def identify_closed_contours(skeleton, min_area=100):
    """
    Identifica i percorsi chiusi (cicli) nello scheletro.

    Un percorso è chiuso se esiste un ciclo: da un pixel puoi tornare
    allo stesso pixel seguendo i pixel connessi dello scheletro.

    Approccio: Le regioni di sfondo completamente circondate dallo scheletro
    (che NON toccano i bordi) sono all'interno di cicli chiusi.

    Args:
        skeleton: Scheletro binario (linee 1 pixel)
        min_area: Area minima per considerare una regione chiusa

    Returns:
        filled_mask: Maschera con cicli chiusi riempiti
        stats: Statistiche sui percorsi
    """
    print("Identificazione cicli chiusi nello scheletro...")

    # Inverti: scheletro diventa nero, sfondo diventa bianco
    inverted = 255 - skeleton

    # Etichetta componenti connesse dello sfondo
    labeled_bg = measure.label(inverted, connectivity=2)

    print(f"Trovate {labeled_bg.max()} componenti connesse nello sfondo")

    # Identifica quali componenti toccano il bordo
    h, w = skeleton.shape
    border_labels = set()

    # Bordi superiore e inferiore
    border_labels.update(labeled_bg[0, :])
    border_labels.update(labeled_bg[-1, :])

    # Bordi sinistro e destro
    border_labels.update(labeled_bg[:, 0])
    border_labels.update(labeled_bg[:, -1])

    # Rimuovi lo zero (sfondo)
    border_labels.discard(0)

    print(f"Componenti che toccano il bordo: {len(border_labels)}")

    # Crea maschera per cicli chiusi
    filled_mask = np.zeros_like(skeleton)

    closed_count = 0
    closed_areas = []

    # Per ogni componente che NON tocca il bordo
    for region in measure.regionprops(labeled_bg):
        label = region.label
        area = region.area

        # Salta se tocca il bordo o è troppo piccola
        if label in border_labels or area < min_area:
            continue

        # Questa è una regione chiusa!
        filled_mask[labeled_bg == label] = 255
        closed_count += 1
        closed_areas.append(area)

    stats = {
        'total_components': labeled_bg.max(),
        'border_components': len(border_labels),
        'closed': closed_count,
        'open': labeled_bg.max() - len(border_labels),
        'avg_closed_area': np.mean(closed_areas) if closed_areas else 0,
        'min_closed_area': np.min(closed_areas) if closed_areas else 0,
        'max_closed_area': np.max(closed_areas) if closed_areas else 0
    }

    print(f"Cicli CHIUSI trovati: {closed_count}")
    print(f"Percorsi APERTI: {stats['open']}")

    return filled_mask, stats


def analyze_fiber_contours(binary_mask, min_contour_area=100, border_exclusion=0):
    """
    Analizza i contorni nella maschera per identificare quali sono chiusi.

    Args:
        binary_mask: Maschera binaria con contorni delle fibre
        min_contour_area: Area minima per considerare un contorno
        border_exclusion: Larghezza del bordo da escludere (in pixel)

    Returns:
        closed_mask, open_mask, stats
    """
    print("Ricerca contorni...")

    # Se richiesto, escludi il bordo
    mask_to_analyze = binary_mask.copy()
    if border_exclusion > 0:
        print(f"Esclusione bordo di {border_exclusion} pixel...")
        # Crea maschera senza bordi
        h, w = binary_mask.shape
        mask_to_analyze[:border_exclusion, :] = 0  # Bordo superiore
        mask_to_analyze[-border_exclusion:, :] = 0  # Bordo inferiore
        mask_to_analyze[:, :border_exclusion] = 0  # Bordo sinistro
        mask_to_analyze[:, -border_exclusion:] = 0  # Bordo destro

    # Trova contorni usando OpenCV
    contours, hierarchy = cv2.findContours(
        mask_to_analyze,
        cv2.RETR_EXTERNAL,  # Solo contorni esterni
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Trovati {len(contours)} contorni (dopo esclusione bordi)")

    closed_mask = np.zeros_like(binary_mask)
    open_mask = np.zeros_like(binary_mask)

    closed_count = 0
    open_count = 0
    closed_contours_list = []
    open_contours_list = []

    print("Analisi contorni...")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue

        # Calcola il perimetro
        perimeter = cv2.arcLength(contour, closed=True)

        # Verifica se il contorno è chiuso analizzando la "compattezza"
        # Contorni chiusi hanno circolarità vicina a 1
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # Un contorno è considerato "chiuso" se ha buona circolarità
        # e l'area interna è ben definita
        if circularity > 0.3:  # Soglia empirica
            # Contorno chiuso - riempi l'interno
            cv2.drawContours(closed_mask, [contour], -1, 255, thickness=cv2.FILLED)
            closed_count += 1
            closed_contours_list.append(contour)
        else:
            # Contorno aperto o molto irregolare
            cv2.drawContours(open_mask, [contour], -1, 255, thickness=2)
            open_count += 1
            open_contours_list.append(contour)

    stats = {
        'total': len(contours),
        'closed': closed_count,
        'open': open_count,
        'closed_contours': closed_contours_list,
        'open_contours': open_contours_list
    }

    return closed_mask, open_mask, stats


def close_open_contours(binary_mask, open_contours, max_gap=10):
    """
    Prova a chiudere i contorni aperti colmando i gap.

    Args:
        binary_mask: Maschera originale
        open_contours: Lista di contorni aperti
        max_gap: Distanza massima di gap da chiudere

    Returns:
        Maschera con contorni chiusi
    """
    print(f"Chiusura contorni aperti (max gap: {max_gap} pixel)...")

    result = binary_mask.copy()

    for contour in open_contours:
        # Prova con dilation morfologica per chiudere piccoli gap
        mask_single = np.zeros_like(binary_mask)
        cv2.drawContours(mask_single, [contour], -1, 255, thickness=2)

        # Dilata per chiudere gap piccoli
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
        dilated = cv2.dilate(mask_single, kernel, iterations=1)

        # Poi erodi per ritornare alla dimensione originale
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Riempi l'area interna
        closed_filled = ndi.binary_fill_holes(eroded > 0).astype(np.uint8) * 255

        # Aggiungi al risultato
        result = cv2.bitwise_or(result, closed_filled)

    return result


def visualize_skeleton(original, cleaned, skeleton, filled_mask, stats, output_dir, base_name):
    """Visualizza il risultato della scheletonizzazione e identificazione percorsi chiusi."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creazione visualizzazioni...")

    # Crea visualizzazione a colori dello scheletro
    skeleton_colored = np.zeros((*original.shape, 3), dtype=np.uint8)
    skeleton_colored[skeleton > 0] = [255, 255, 255]  # Bianco per lo scheletro

    # Percorsi chiusi in VERDE
    closed_green = np.zeros((*original.shape, 3), dtype=np.uint8)
    closed_green[filled_mask > 0] = [0, 255, 0]

    # Overlay percorsi chiusi su originale
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay[filled_mask > 0] = [0, 255, 0]

    # Visualizzazione principale - 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Originale
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Maschera Originale', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Dopo pulizia
    axes[0, 1].imshow(cleaned, cmap='gray')
    axes[0, 1].set_title('Dopo Pulizia Rumore', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Scheletro
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].set_title('Scheletro (linee 1 pixel)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Percorsi chiusi in verde
    axes[1, 0].imshow(closed_green)
    axes[1, 0].set_title(f'Percorsi Chiusi ({stats["closed"]})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Overlay su originale
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay su Originale', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Statistiche come testo
    stats_text = f"""STATISTICHE:

Totale contorni: {stats['total']}
Percorsi CHIUSI: {stats['closed']}
Percorsi APERTI: {stats['open']}

Area media chiusi: {stats['avg_closed_area']:.0f} px
Area min: {stats['min_closed_area']:.0f} px
Area max: {stats['max_closed_area']:.0f} px
"""
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Salva
    vis_path = output_dir / f"{base_name}_skeleton_analysis.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualizzazione salvata in: {vis_path}")
    plt.close()

    # Salva maschere separate
    skeleton_path = output_dir / f"{base_name}_skeleton.png"
    cv2.imwrite(str(skeleton_path), skeleton)
    print(f"Scheletro salvato in: {skeleton_path}")

    cleaned_path = output_dir / f"{base_name}_cleaned.png"
    cv2.imwrite(str(cleaned_path), cleaned)
    print(f"Maschera pulita salvata in: {cleaned_path}")


def visualize_classification(original, closed_mask, open_mask, stats, output_dir, base_name):
    """Visualizza la classificazione dei contorni (chiusi vs aperti)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creazione visualizzazioni...")

    # Crea immagine colorata
    colored = np.zeros((*original.shape, 3), dtype=np.uint8)

    # Contorni originali in bianco
    colored[original > 0] = [255, 255, 255]

    # Contorni chiusi riempiti di VERDE
    colored[closed_mask > 0] = [0, 255, 0]

    # Contorni aperti in ROSSO
    colored[open_mask > 0] = [255, 0, 0]

    # Visualizzazione principale - 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Originale
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Maschera Originale (Contorni)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Classificazione colorata
    axes[1].imshow(colored)
    axes[1].set_title(f'Verde=Chiusi ({stats["closed"]}), Rosso=Aperti ({stats["open"]})',
                        fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Solo contorni chiusi (riempiti di verde)
    green_only = np.zeros((*original.shape, 3), dtype=np.uint8)
    green_only[closed_mask > 0] = [0, 255, 0]
    axes[2].imshow(green_only)
    axes[2].set_title(f'Solo Contorni Chiusi Riempiti ({stats["closed"]})',
                        fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # Salva
    vis_path = output_dir / f"{base_name}_contour_analysis.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualizzazione salvata in: {vis_path}")
    plt.close()

    # Salva maschere separate
    closed_path = output_dir / f"{base_name}_closed_filled.png"
    cv2.imwrite(str(closed_path), closed_mask)
    print(f"Contorni chiusi riempiti salvati in: {closed_path}")

    colored_path = output_dir / f"{base_name}_colored_classification.png"
    cv2.imwrite(str(colored_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    print(f"Classificazione colorata salvata in: {colored_path}")


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Analizza contorni di fibre muscolari in sezione'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Percorso alla maschera binaria con contorni'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Directory di output (default: output)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=500,
        help='Area minima per rimuovere rumore (default: 500 pixel)'
    )
    parser.add_argument(
        '--exclude-border',
        type=int,
        default=50,
        help='Larghezza del bordo da escludere dall\'analisi (default: 50 pixel)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SCHELETONIZZAZIONE CONTORNI FIBRE MUSCOLARI")
    print("=" * 60)

    # Carica
    print(f"\nCaricamento: {args.input}")
    original = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Errore: Impossibile caricare: {args.input}")
        return

    print(f"Dimensioni: {original.shape}")

    # Pulisci e scheletonizza
    print("\nPulizia e scheletonizzazione...")
    skeleton, cleaned = clean_and_skeletonize(
        original,
        min_area=args.min_area,
        border_exclusion=args.exclude_border
    )

    # Identifica percorsi chiusi
    print("\nIdentificazione percorsi chiusi...")
    filled_mask, stats = identify_closed_contours(
        skeleton,
        min_area=100
    )

    # Salva risultati
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input).stem

    # Visualizza
    print("\nCreazione visualizzazioni...")
    visualize_skeleton(original, cleaned, skeleton, filled_mask, stats, args.output, base_name)

    # Salva maschera percorsi chiusi
    filled_path = output_dir / f"{base_name}_closed_filled.png"
    cv2.imwrite(str(filled_path), filled_mask)
    print(f"Percorsi chiusi salvati in: {filled_path}")

    print("\n" + "=" * 60)
    print("COMPLETATO")
    print("=" * 60)
    print(f"Percorsi CHIUSI: {stats['closed']}")
    print(f"Percorsi APERTI: {stats['open']}")
    print("=" * 60)

    print("\nANALISI COMPLETATA!")


if __name__ == '__main__':
    main()
