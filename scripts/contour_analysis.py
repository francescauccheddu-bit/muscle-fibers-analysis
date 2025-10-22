#!/usr/bin/env python3
"""
Script per analisi dettagliata dei contorni delle fibre muscolari.

Questo script permette di:
- Visualizzare i contorni di ogni fibra
- Identificare contorni aperti vs chiusi
- Esaminare la qualità della segmentazione
- Salvare immagini ad alta risoluzione dei bordi
- Salvare immagine binaria con solo fibre a contorni chiusi
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import filters, measure, morphology
from skimage.segmentation import find_boundaries


def load_and_segment(image_path, threshold_method='otsu'):
    """Carica e segmenta l'immagine."""
    print(f"Caricamento: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    
    print("Preprocessing...")
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    preprocessed = cv2.equalizeHist(blurred)
    
    print("Segmentazione...")
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(preprocessed)
        binary = preprocessed > threshold
    else:
        binary = filters.threshold_local(preprocessed, block_size=35)
        binary = preprocessed > binary
    
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    segmented = measure.label(binary)
    
    print(f"Fibre identificate: {segmented.max()}")
    return image, segmented


def analyze_contours(segmented):
    """Analizza i contorni per identificare quelli aperti."""
    print("Analisi proprietà fibre...")
    regions = measure.regionprops(segmented)
    
    open_contours = []
    closed_contours = []
    
    for region in regions:
        # Un contorno è considerato "aperto" se tocca i bordi dell'immagine
        minr, minc, maxr, maxc = region.bbox
        touches_border = (minr == 0 or minc == 0 or 
                         maxr == segmented.shape[0] or 
                         maxc == segmented.shape[1])
        
        # Calcola la circolarità (1.0 = cerchio perfetto, < 1.0 = meno circolare)
        circularity = 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
        
        contour_info = {
            'label': region.label,
            'area': region.area,
            'perimeter': region.perimeter,
            'circularity': circularity,
            'touches_border': touches_border,
            'bbox': region.bbox
        }
        
        if touches_border or circularity < 0.3:
            open_contours.append(contour_info)
        else:
            closed_contours.append(contour_info)
    
    return open_contours, closed_contours


def create_closed_fibers_binary(segmented, closed_contours):
    """Crea immagine binaria con solo le fibre a contorni chiusi."""
    print("Creazione immagine binaria fibre chiuse...")
    closed_binary = np.zeros(segmented.shape, dtype=np.uint8)
    
    for contour in closed_contours:
        label = contour['label']
        closed_binary[segmented == label] = 255
    
    return closed_binary


def create_visualizations(image, segmented, open_contours, closed_contours, output_dir, base_name):
    """Crea diverse visualizzazioni dei contorni."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crea immagine binaria con solo fibre chiuse
    closed_binary = create_closed_fibers_binary(segmented, closed_contours)
    
    # Calcola boundaries UNA VOLTA SOLA per tutta l'immagine
    print("Calcolo boundaries...")
    boundaries = find_boundaries(segmented, mode='outer')
    
    # Crea maschera per i contorni aperti
    print("Identificazione contorni aperti/chiusi...")
    open_labels = set([c['label'] for c in open_contours])
    
    # Crea mappa per identificare quali boundaries appartengono a fibre aperte
    open_boundary_mask = np.zeros(segmented.shape, dtype=bool)
    for label in open_labels:
        # Dilata leggermente la maschera della fibra per catturare il boundary
        fiber_mask = segmented == label
        dilated = morphology.dilation(fiber_mask, morphology.disk(1))
        open_boundary_mask |= (dilated & boundaries)
    
    # 1. Visualizzazione completa con contorni colorati
    print("Creazione visualizzazioni...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # Originale
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Immagine Originale', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Contorni chiusi (verde) vs aperti (rosso)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Prima colora tutti i bordi in verde
    image_rgb[boundaries & ~open_boundary_mask] = [0, 255, 0]
    # Poi colora in rosso i bordi aperti
    image_rgb[open_boundary_mask] = [255, 0, 0]
    
    axes[0, 1].imshow(image_rgb)
    axes[0, 1].set_title(f'Contorni: Verde=Chiusi ({len(closed_contours)}), Rosso=Aperti ({len(open_contours)})', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Solo contorni su sfondo bianco
    contours_only = np.ones((*segmented.shape, 3), dtype=np.uint8) * 255
    contours_only[boundaries & ~open_boundary_mask] = [0, 255, 0]
    contours_only[open_boundary_mask] = [255, 0, 0]
    
    axes[0, 2].imshow(contours_only)
    axes[0, 2].set_title('Solo Contorni su Sfondo Bianco', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Immagine binaria con solo fibre chiuse
    axes[1, 0].imshow(closed_binary, cmap='gray')
    axes[1, 0].set_title(f'Binario: Solo Fibre Chiuse ({len(closed_contours)} fibre)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Fibre chiuse sovrapposte all'originale
    closed_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    closed_overlay[closed_binary == 255] = [0, 255, 0]
    axes[1, 1].imshow(closed_overlay)
    axes[1, 1].set_title('Fibre Chiuse (verde) su Originale', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Distribuzione circolarità
    all_circularities = [c['circularity'] for c in closed_contours + open_contours]
    axes[1, 2].hist(all_circularities, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(x=0.3, color='r', linestyle='--', label='Soglia apertura (0.3)')
    axes[1, 2].set_title('Distribuzione Circolarità Fibre', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Circolarità (1.0 = cerchio perfetto)')
    axes[1, 2].set_ylabel('Frequenza')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva figura principale
    print("Salvataggio visualizzazione principale...")
    main_path = output_dir / f"{base_name}_contour_analysis.png"
    plt.savefig(main_path, dpi=300, bbox_inches='tight')
    print(f"Analisi contorni salvata in: {main_path}")
    plt.close()
    
    # 2. Salva immagine binaria con solo fibre chiuse
    binary_path = output_dir / f"{base_name}_closed_fibers_binary.png"
    cv2.imwrite(str(binary_path), closed_binary)
    print(f"Immagine binaria fibre chiuse salvata in: {binary_path}")
    
    # 3. Salva immagine ad alta risoluzione solo con bordi colorati
    high_res_path = output_dir / f"{base_name}_contours_highres.png"
    cv2.imwrite(str(high_res_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Immagine ad alta risoluzione salvata in: {high_res_path}")


def save_contour_report(open_contours, closed_contours, output_dir, base_name):
    """Salva un report CSV con i dettagli dei contorni."""
    output_dir = Path(output_dir)
    
    all_contours = []
    for c in closed_contours:
        c_copy = c.copy()
        c_copy['status'] = 'closed'
        c_copy.pop('bbox', None)
        all_contours.append(c_copy)
    for c in open_contours:
        c_copy = c.copy()
        c_copy['status'] = 'open'
        c_copy.pop('bbox', None)
        all_contours.append(c_copy)
    
    df = pd.DataFrame(all_contours)
    df = df.sort_values('area', ascending=False)
    
    csv_path = output_dir / f"{base_name}_contour_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"Report contorni salvato in: {csv_path}")
    
    # Stampa statistiche
    print("\n" + "=" * 60)
    print("STATISTICHE CONTORNI")
    print("=" * 60)
    total = len(closed_contours) + len(open_contours)
    print(f"Contorni chiusi: {len(closed_contours)} ({len(closed_contours)/total*100:.1f}%)")
    print(f"Contorni aperti: {len(open_contours)} ({len(open_contours)/total*100:.1f}%)")
    print(f"\nCircolarità media (chiusi): {np.mean([c['circularity'] for c in closed_contours]):.3f}")
    if open_contours:
        print(f"Circolarità media (aperti): {np.mean([c['circularity'] for c in open_contours]):.3f}")
    print("=" * 60)


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Analisi dettagliata dei contorni delle fibre muscolari'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Percorso all\'immagine di input'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Directory di output (default: output)'
    )
    parser.add_argument(
        '--threshold',
        type=str,
        default='otsu',
        choices=['otsu', 'adaptive'],
        help='Metodo di thresholding (default: otsu)'
    )
    
    args = parser.parse_args()
    
    # Carica e segmenta
    image, segmented = load_and_segment(args.input, args.threshold)
    
    # Analizza contorni
    print("\nAnalisi contorni...")
    open_contours, closed_contours = analyze_contours(segmented)
    
    # Crea visualizzazioni
    base_name = Path(args.input).stem
    print("\nCreazione visualizzazioni...")
    create_visualizations(image, segmented, open_contours, closed_contours, args.output, base_name)
    
    # Salva report
    print("\nCreazione report...")
    save_contour_report(open_contours, closed_contours, args.output, base_name)
    
    print("\n" + "=" * 60)
    print("ANALISI COMPLETATA!")
    print("=" * 60)


if __name__ == '__main__':
    main()
