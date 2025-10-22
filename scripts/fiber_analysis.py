#!/usr/bin/env python3
"""
Script per l'analisi di immagini al microscopio di fibre muscolari.

Questo script esegue:
1. Caricamento dell'immagine
2. Preprocessing (riduzione rumore, normalizzazione)
3. Segmentazione delle fibre
4. Analisi morfometrica (area, perimetro, diametri)
5. Visualizzazione e salvataggio risultati
"""

import argparse
import os
from pathlib import Path
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import filters, measure, morphology, segmentation
from skimage.color import label2rgb


class MuscleAnalyzer:
    """Classe per l'analisi di immagini di fibre muscolari."""

    def __init__(self, image_path):
        """
        Inizializza l'analizzatore.

        Args:
            image_path: Percorso all'immagine da analizzare
        """
        self.image_path = image_path
        self.image = None
        self.preprocessed = None
        self.segmented = None
        self.results = None

    def load_image(self):
        """Carica l'immagine in grayscale."""
        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Impossibile caricare l'immagine: {self.image_path}")
        print(f"Immagine caricata: {self.image.shape}")
        return self

    def preprocess(self):
        """Applica preprocessing all'immagine."""
        blurred = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.preprocessed = cv2.equalizeHist(blurred)
        print("Preprocessing completato")
        return self

    def segment_fibers(self, threshold_method='otsu'):
        """
        Segmenta le fibre muscolari nell'immagine.

        Args:
            threshold_method: Metodo di thresholding ('otsu', 'adaptive')
        """
        if threshold_method == 'otsu':
            threshold = filters.threshold_otsu(self.preprocessed)
            binary = self.preprocessed > threshold
        elif threshold_method == 'adaptive':
            binary = filters.threshold_local(self.preprocessed, block_size=35)
            binary = self.preprocessed > binary
        else:
            raise ValueError(f"Metodo non supportato: {threshold_method}")

        binary = morphology.remove_small_objects(binary, min_size=50)
        binary = morphology.remove_small_holes(binary, area_threshold=50)
        self.segmented = measure.label(binary)

        n_fibers = self.segmented.max()
        print(f"Segmentazione completata: {n_fibers} fibre identificate")
        return self

    def analyze_fibers(self):
        """Estrae parametri morfometrici per ogni fibra."""
        props = measure.regionprops_table(
            self.segmented,
            intensity_image=self.image,
            properties=(
                'label',
                'area',
                'perimeter',
                'eccentricity',
                'equivalent_diameter',
                'major_axis_length',
                'minor_axis_length',
                'mean_intensity',
                'centroid'
            )
        )

        self.results = pd.DataFrame(props)

        self.results['circularity'] = (
            4 * np.pi * self.results['area'] / (self.results['perimeter'] ** 2)
        )
        self.results['aspect_ratio'] = (
            self.results['major_axis_length'] / self.results['minor_axis_length']
        )

        print(f"Analisi completata su {len(self.results)} fibre")
        print("\nStatistiche descrittive:")
        print(self.results[['area', 'perimeter', 'equivalent_diameter']].describe())

        return self

    def save_results(self, output_dir):
        """
        Salva i risultati dell'analisi.

        Args:
            output_dir: Directory di output
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(self.image_path).stem

        csv_path = output_dir / f"{base_name}_results.csv"
        self.results.to_csv(csv_path, index=False)
        print(f"Risultati salvati in: {csv_path}")

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        axes[0, 0].imshow(self.image, cmap='gray')
        axes[0, 0].set_title('Immagine Originale')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.preprocessed, cmap='gray')
        axes[0, 1].set_title('Preprocessing')
        axes[0, 1].axis('off')

        image_label_overlay = label2rgb(self.segmented, image=self.image, bg_label=0)
        axes[1, 0].imshow(image_label_overlay)
        axes[1, 0].set_title(f'Segmentazione ({self.segmented.max()} fibre)')
        axes[1, 0].axis('off')

        axes[1, 1].hist(self.results['area'], bins=30, edgecolor='black')
        axes[1, 1].set_title('Distribuzione Aree Fibre')
        axes[1, 1].set_xlabel('Area (pixel²)')
        axes[1, 1].set_ylabel('Frequenza')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = output_dir / f"{base_name}_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Visualizzazione salvata in: {fig_path}")
        plt.close()

        return self


def find_images_in_data_folder():
    """Trova tutte le immagini nella cartella data/."""
    data_dir = Path('data')
    if not data_dir.exists():
        print("Errore: La cartella 'data' non esiste!")
        return []
    
    extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
    images = []
    
    for ext in extensions:
        images.extend(data_dir.glob(ext))
    
    images = [img for img in images if img.name != '.gitkeep']
    
    # Rimuovi duplicati basandosi sul percorso assoluto risolto
    unique_images = {}
    for img in images:
        resolved = img.resolve()
        unique_images[str(resolved).lower()] = img
    
    return sorted(unique_images.values())


def process_single_image(image_path, output_dir, threshold_method):
    """Processa una singola immagine."""
    print("\n" + "=" * 60)
    print(f"ELABORAZIONE: {Path(image_path).name}")
    print("=" * 60)
    
    try:
        analyzer = MuscleAnalyzer(image_path)
        analyzer.load_image().preprocess().segment_fibers(threshold_method=threshold_method).analyze_fibers().save_results(output_dir)
        print("Completato con successo!")
        return True
    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
        return False


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Analisi di immagini al microscopio di fibre muscolari'
    )
    parser.add_argument(
        '--input',
        type=str,
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
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Processa tutte le immagini nella cartella data/'
    )

    args = parser.parse_args()

    if args.batch:
        print("=" * 60)
        print("MODALITA BATCH: Ricerca immagini in 'data/'")
        print("=" * 60)
        
        images = find_images_in_data_folder()
        
        if not images:
            print("\nNessuna immagine trovata nella cartella 'data/'")
            print("Formati supportati: .tif, .tiff, .png, .jpg, .jpeg")
            return
        
        print(f"\nTrovate {len(images)} immagini da processare:\n")
        for img in images:
            print(f"  - {img.name}")
        print()
        
        success_count = 0
        fail_count = 0
        
        for image_path in images:
            if process_single_image(image_path, args.output, args.threshold):
                success_count += 1
            else:
                fail_count += 1
        
        print("\n" + "=" * 60)
        print("RIEPILOGO BATCH")
        print("=" * 60)
        print(f"Totale immagini: {len(images)}")
        print(f"Elaborate con successo: {success_count}")
        print(f"Errori: {fail_count}")
        print("=" * 60)
    
    else:
        if not args.input:
            print("Errore: Devi specificare --input oppure usare --batch")
            print("\nEsempi d'uso:")
            print("  python scripts/fiber_analysis.py --input data/immagine.tif --output output/")
            print("  python scripts/fiber_analysis.py --batch --output output/")
            return
        
        if not os.path.exists(args.input):
            print(f"Errore: File non trovato: {args.input}")
            return

        process_single_image(args.input, args.output, args.threshold)


if __name__ == '__main__':
    main()
