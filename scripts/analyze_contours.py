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

    Trova solo i cicli PIU' INTERNI - quelli che non contengono altri cicli.

    Approccio:
    1. Trova contorni dello sfondo con gerarchia
    2. Identifica contorni che non hanno figli (leaf nodes)
    3. Esclude contorni che toccano i bordi
    4. Riempie solo i cicli più interni

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

    # Trova contorni con gerarchia
    # hierarchy: [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(
        inverted,
        cv2.RETR_TREE,  # Ottieni gerarchia completa
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Trovati {len(contours)} contorni totali")

    if len(contours) == 0 or hierarchy is None:
        return np.zeros_like(skeleton), {
            'total_contours': 0,
            'closed': 0,
            'open': 0,
            'avg_closed_area': 0,
            'min_closed_area': 0,
            'max_closed_area': 0
        }

    hierarchy = hierarchy[0]  # Rimuovi dimensione extra
    h, w = skeleton.shape

    # Crea maschere per cicli chiusi
    filled_mask = np.zeros_like(skeleton)  # Aree riempite
    cycle_lines_mask = np.zeros_like(skeleton)  # Solo le linee dello scheletro nei cicli

    closed_count = 0
    closed_areas = []
    skipped_border = 0
    skipped_parent = 0
    skipped_small = 0

    # Analizza ogni contorno
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        # Controlla se ha figli (First_Child != -1)
        has_children = hierarchy[idx][2] != -1
        if has_children:
            skipped_parent += 1
            continue

        # Controlla se tocca il bordo
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]
        touches_border = (
            np.any(x_coords == 0) or np.any(x_coords == w - 1) or
            np.any(y_coords == 0) or np.any(y_coords == h - 1)
        )

        if touches_border:
            skipped_border += 1
            continue

        # Controlla area minima
        if area < min_area:
            skipped_small += 1
            continue

        # Questo è un ciclo chiuso INTERNO!
        print(f"  Ciclo chiuso {closed_count + 1}: area={area:.0f} px")

        # Riempi l'area interna
        cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Per trovare le linee dello scheletro che formano il ciclo:
        # Dilata leggermente l'area riempita e fai AND con lo scheletro
        temp_filled = np.zeros_like(skeleton)
        cv2.drawContours(temp_filled, [contour], -1, 255, thickness=cv2.FILLED)
        dilated = cv2.dilate(temp_filled, np.ones((3, 3), np.uint8), iterations=1)
        cycle_skeleton = cv2.bitwise_and(skeleton, dilated)
        cycle_lines_mask = cv2.bitwise_or(cycle_lines_mask, cycle_skeleton)

        closed_count += 1
        closed_areas.append(area)

    print(f"\nRisultati:")
    print(f"  Cicli chiusi interni: {closed_count}")
    print(f"  Esclusi (hanno figli): {skipped_parent}")
    print(f"  Esclusi (toccano bordi): {skipped_border}")
    print(f"  Esclusi (troppo piccoli): {skipped_small}")

    stats = {
        'total_contours': len(contours),
        'closed': closed_count,
        'open': len(contours) - closed_count,
        'avg_closed_area': np.mean(closed_areas) if closed_areas else 0,
        'min_closed_area': np.min(closed_areas) if closed_areas else 0,
        'max_closed_area': np.max(closed_areas) if closed_areas else 0
    }

    return filled_mask, cycle_lines_mask, stats


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


def visualize_skeleton(original, cleaned, skeleton, cycle_lines_mask, filled_mask, stats, output_dir, base_name):
    """Visualizza il risultato della scheletonizzazione e identificazione percorsi chiusi."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creazione visualizzazioni...")

    # Aree riempite in verde
    filled_green = np.zeros((*original.shape, 3), dtype=np.uint8)
    filled_green[filled_mask > 0] = [0, 255, 0]  # Aree chiuse riempite in VERDE

    # Overlay su originale: aree riempite in verde
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay[filled_mask > 0] = [0, 255, 0]  # Aree chiuse in VERDE

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

    # Aree riempite in verde
    axes[1, 0].imshow(filled_green)
    axes[1, 0].set_title(f'Cicli Chiusi Riempiti ({stats["closed"]})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Overlay su originale
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay: Aree Cicli Chiusi in Verde', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Statistiche come testo
    stats_text = f"""STATISTICHE:

Contorni totali: {stats['total_contours']}
Cicli CHIUSI (interni): {stats['closed']}
Altri contorni: {stats['open']}

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

    # Salva linee dei cicli chiusi
    cycle_lines_path = output_dir / f"{base_name}_cycle_lines.png"
    cv2.imwrite(str(cycle_lines_path), cycle_lines_mask)
    print(f"Linee cicli chiusi salvate in: {cycle_lines_path}")

    # Salva aree riempite
    filled_path = output_dir / f"{base_name}_closed_filled.png"
    cv2.imwrite(str(filled_path), filled_mask)
    print(f"Aree chiuse riempite salvate in: {filled_path}")


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
    parser.add_argument(
        '--min-cycle-area',
        type=int,
        default=50,
        help='Area minima per considerare un ciclo chiuso (default: 50 pixel)'
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
    filled_mask, cycle_lines_mask, stats = identify_closed_contours(
        skeleton,
        min_area=args.min_cycle_area
    )

    # Salva risultati
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input).stem

    # Visualizza
    print("\nCreazione visualizzazioni...")
    visualize_skeleton(original, cleaned, skeleton, cycle_lines_mask, filled_mask, stats, args.output, base_name)

    print("\n" + "=" * 60)
    print("COMPLETATO")
    print("=" * 60)
    print(f"Percorsi CHIUSI: {stats['closed']}")
    print(f"Percorsi APERTI: {stats['open']}")
    print("=" * 60)

    print("\nANALISI COMPLETATA!")


if __name__ == '__main__':
    main()
