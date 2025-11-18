#!/usr/bin/env python3
"""
Script per confrontare risultati di multiple analisi su immagini diverse.
Genera un report comparativo dettagliato con statistiche e visualizzazioni.
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_analysis_results(output_dir):
    """
    Carica i risultati di un'analisi da una directory di output.

    Args:
        output_dir: Path alla directory con metadata.json, summary_statistics.csv, fibers_statistics.csv

    Returns:
        dict con 'metadata', 'summary', 'fibers_df'
    """
    output_path = Path(output_dir)

    # Carica metadata
    metadata_file = output_path / 'metadata.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata non trovato: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Carica summary statistics
    summary_file = output_path / 'summary_statistics.csv'
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary statistics non trovato: {summary_file}")

    summary = pd.read_csv(summary_file)

    # Carica fibers statistics
    fibers_file = output_path / 'fibers_statistics.csv'
    if not fibers_file.exists():
        raise FileNotFoundError(f"Fibers statistics non trovato: {fibers_file}")

    fibers_df = pd.read_csv(fibers_file)

    return {
        'metadata': metadata,
        'summary': summary,
        'fibers_df': fibers_df,
        'output_dir': str(output_path)
    }

def create_comparison_visualizations(analyses, labels, output_dir):
    """
    Crea visualizzazioni comparative tra le analisi.

    Args:
        analyses: Lista di dict con risultati analisi
        labels: Lista di label per ogni analisi
        output_dir: Directory dove salvare le visualizzazioni
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Confronto distribuzione aree
    print("\nCreazione visualizzazioni comparative...")

    fig, axes = plt.subplots(len(analyses), 1, figsize=(12, 4 * len(analyses)))
    if len(analyses) == 1:
        axes = [axes]

    for idx, (analysis, label) in enumerate(zip(analyses, labels)):
        fibers_df = analysis['fibers_df']

        # Determina se abbiamo dati calibrati (µm²) o solo pixel
        if 'area_um2' in fibers_df.columns:
            areas = fibers_df['area_um2'].values
            unit = 'µm²'
        else:
            areas = fibers_df['area_px'].values
            unit = 'px²'

        ax = axes[idx]
        ax.hist(areas, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'Area Fibra ({unit})')
        ax.set_ylabel('Frequenza')
        ax.set_title(f'{label}\nN = {len(areas)} fibre | Media = {np.mean(areas):.1f} {unit}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'comparison_area_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Salvato: comparison_area_distributions.png")
    plt.close()

    # 2. Box plot comparativo
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    for analysis in analyses:
        fibers_df = analysis['fibers_df']
        if 'area_um2' in fibers_df.columns:
            data_to_plot.append(fibers_df['area_um2'].values)
            unit = 'µm²'
        else:
            data_to_plot.append(fibers_df['area_px'].values)
            unit = 'px²'

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Colora i box
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(analyses)]):
        patch.set_facecolor(color)

    ax.set_ylabel(f'Area Fibra ({unit})')
    ax.set_title('Confronto Distribuzione Aree Fibre')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'comparison_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"  Salvato: comparison_boxplot.png")
    plt.close()

    # 3. Grafico a barre per metriche principali
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Estrai metriche
    n_fibers = [analysis['metadata']['n_fibers'] for analysis in analyses]
    mean_areas = []
    median_areas = []
    coverage = [analysis['metadata']['mask_coverage_percent'] for analysis in analyses]

    for analysis in analyses:
        summary = analysis['summary']
        if 'mean_area_um2' in summary['metric'].values:
            mean_areas.append(summary[summary['metric'] == 'mean_area_um2']['value'].values[0])
            median_areas.append(summary[summary['metric'] == 'median_area_um2']['value'].values[0])
            area_unit = 'µm²'
        else:
            mean_areas.append(summary[summary['metric'] == 'mean_area_px']['value'].values[0])
            median_areas.append(summary[summary['metric'] == 'median_area_px']['value'].values[0])
            area_unit = 'px²'

    # Plot numero fibre
    axes[0, 0].bar(labels, n_fibers, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(analyses)])
    axes[0, 0].set_ylabel('Numero Fibre')
    axes[0, 0].set_title('Numero Totale Fibre Identificate')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(n_fibers):
        axes[0, 0].text(i, v, f'{v:,.0f}', ha='center', va='bottom')

    # Plot area media
    axes[0, 1].bar(labels, mean_areas, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(analyses)])
    axes[0, 1].set_ylabel(f'Area Media ({area_unit})')
    axes[0, 1].set_title('Area Media Fibre')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mean_areas):
        axes[0, 1].text(i, v, f'{v:,.1f}', ha='center', va='bottom')

    # Plot area mediana
    axes[1, 0].bar(labels, median_areas, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(analyses)])
    axes[1, 0].set_ylabel(f'Area Mediana ({area_unit})')
    axes[1, 0].set_title('Area Mediana Fibre')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(median_areas):
        axes[1, 0].text(i, v, f'{v:,.1f}', ha='center', va='bottom')

    # Plot coverage
    axes[1, 1].bar(labels, coverage, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'][:len(analyses)])
    axes[1, 1].set_ylabel('Coverage (%)')
    axes[1, 1].set_title('Copertura Maschera')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(coverage):
        axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path / 'comparison_metrics.png', dpi=300, bbox_inches='tight')
    print(f"  Salvato: comparison_metrics.png")
    plt.close()

def generate_comparison_report(analyses, labels, output_file):
    """
    Genera un report markdown con confronto dettagliato.

    Args:
        analyses: Lista di dict con risultati analisi
        labels: Lista di label per ogni analisi
        output_file: Path al file markdown di output
    """
    print(f"\nGenerazione report: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Report Comparativo: Analisi Fibre Muscolari\n\n")
        f.write(f"**Data**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("---\n\n")

        # Tabella sommaria
        f.write("## 📊 Tabella Comparativa Principale\n\n")
        f.write("| Metrica | " + " | ".join(labels) + " |\n")
        f.write("|---------|" + "|".join(["---"] * len(labels)) + "|\n")

        # Numero fibre
        n_fibers = [analysis['metadata']['n_fibers'] for analysis in analyses]
        f.write(f"| **Numero Fibre** | " + " | ".join([f"**{n:,}**" for n in n_fibers]) + " |\n")

        # Dimensioni immagine
        widths = [analysis['metadata']['image_dimensions']['width'] for analysis in analyses]
        heights = [analysis['metadata']['image_dimensions']['height'] for analysis in analyses]
        f.write(f"| **Dimensioni Immagine** | " + " | ".join([f"{w:,} × {h:,} px" for w, h in zip(widths, heights)]) + " |\n")

        # Coverage
        coverage = [analysis['metadata']['mask_coverage_percent'] for analysis in analyses]
        f.write(f"| **Coverage Maschera** | " + " | ".join([f"{c:.2f}%" for c in coverage]) + " |\n")

        f.write("\n### Statistiche Aree Fibre\n\n")

        # Determina l'unità di misura
        first_summary = analyses[0]['summary']
        if 'mean_area_um2' in first_summary['metric'].values:
            area_metrics = ['mean_area_um2', 'median_area_um2', 'std_area_um2', 'min_area_um2', 'max_area_um2']
            unit = 'µm²'
        else:
            area_metrics = ['mean_area_px', 'median_area_px', 'std_area_px', 'min_area_px', 'max_area_px']
            unit = 'px²'

        metric_names = {
            'mean_area': 'Media',
            'median_area': 'Mediana',
            'std_area': 'Dev. Std.',
            'min_area': 'Minimo',
            'max_area': 'Massimo'
        }

        for metric_key in area_metrics:
            metric_base = metric_key.rsplit('_', 1)[0]  # remove _um2 or _px
            metric_name = metric_names.get(metric_base, metric_base)

            values = []
            for analysis in analyses:
                summary = analysis['summary']
                value = summary[summary['metric'] == metric_key]['value'].values[0]
                values.append(value)

            f.write(f"| **{metric_name}** ({unit}) | " + " | ".join([f"{v:,.2f}" for v in values]) + " |\n")

        # Confronto percentuale
        if len(analyses) >= 2:
            f.write("\n---\n\n")
            f.write(f"## 📈 Variazioni Rispetto a {labels[0]}\n\n")

            baseline = analyses[0]
            baseline_n = baseline['metadata']['n_fibers']
            baseline_summary = baseline['summary']

            if 'mean_area_um2' in baseline_summary['metric'].values:
                baseline_mean = baseline_summary[baseline_summary['metric'] == 'mean_area_um2']['value'].values[0]
            else:
                baseline_mean = baseline_summary[baseline_summary['metric'] == 'mean_area_px']['value'].values[0]

            for i in range(1, len(analyses)):
                comp = analyses[i]
                comp_n = comp['metadata']['n_fibers']
                comp_summary = comp['summary']

                if 'mean_area_um2' in comp_summary['metric'].values:
                    comp_mean = comp_summary[comp_summary['metric'] == 'mean_area_um2']['value'].values[0]
                else:
                    comp_mean = comp_summary[comp_summary['metric'] == 'mean_area_px']['value'].values[0]

                delta_n = comp_n - baseline_n
                delta_n_pct = (delta_n / baseline_n) * 100

                delta_mean = comp_mean - baseline_mean
                delta_mean_pct = (delta_mean / baseline_mean) * 100

                f.write(f"### {labels[i]} vs {labels[0]}\n\n")
                f.write(f"- **Numero Fibre**: {comp_n:,} vs {baseline_n:,} = **{delta_n:+,}** ({delta_n_pct:+.1f}%)\n")
                f.write(f"- **Area Media**: {comp_mean:,.2f} {unit} vs {baseline_mean:,.2f} {unit} = **{delta_mean:+,.2f}** {unit} ({delta_mean_pct:+.1f}%)\n\n")

        # Dettagli per ogni analisi
        f.write("---\n\n")
        f.write("## 🔍 Dettagli Analisi Individuali\n\n")

        for i, (analysis, label) in enumerate(zip(analyses, labels)):
            f.write(f"### {i+1}. {label}\n\n")

            metadata = analysis['metadata']
            summary = analysis['summary']

            f.write(f"**Directory Output**: `{analysis['output_dir']}`\n\n")

            f.write("**Metadati**:\n")
            f.write(f"- Fibre identificate: {metadata['n_fibers']:,}\n")
            f.write(f"- Dimensioni immagine: {metadata['image_dimensions']['width']:,} × {metadata['image_dimensions']['height']:,} px\n")
            f.write(f"- Coverage maschera: {metadata['mask_coverage_percent']:.2f}%\n\n")

            f.write("**Statistiche Aree**:\n")
            for metric_key in area_metrics:
                metric_base = metric_key.rsplit('_', 1)[0]
                metric_name = metric_names.get(metric_base, metric_base)
                value = summary[summary['metric'] == metric_key]['value'].values[0]
                f.write(f"- {metric_name}: {value:,.2f} {unit}\n")
            f.write("\n")

        # Visualizzazioni
        f.write("---\n\n")
        f.write("## 📊 Visualizzazioni\n\n")
        f.write("### Distribuzione Aree\n")
        f.write("![Distribuzione Aree](comparison_area_distributions.png)\n\n")
        f.write("### Box Plot Comparativo\n")
        f.write("![Box Plot](comparison_boxplot.png)\n\n")
        f.write("### Metriche Principali\n")
        f.write("![Metriche](comparison_metrics.png)\n\n")

        # Conclusioni
        f.write("---\n\n")
        f.write("## 💡 Osservazioni\n\n")

        if len(analyses) >= 2:
            # Identifica quale ha più fibre
            max_fibers_idx = np.argmax(n_fibers)
            min_fibers_idx = np.argmin(n_fibers)

            f.write(f"1. **Numero Fibre**: {labels[max_fibers_idx]} ha il maggior numero di fibre identificate ({n_fibers[max_fibers_idx]:,})\n")
            f.write(f"2. **Variazione**: Differenza di {n_fibers[max_fibers_idx] - n_fibers[min_fibers_idx]:,} fibre tra {labels[max_fibers_idx]} e {labels[min_fibers_idx]}\n")

            # Confronta le aree medie
            mean_areas = []
            for analysis in analyses:
                summary = analysis['summary']
                if 'mean_area_um2' in summary['metric'].values:
                    mean_areas.append(summary[summary['metric'] == 'mean_area_um2']['value'].values[0])
                else:
                    mean_areas.append(summary[summary['metric'] == 'mean_area_px']['value'].values[0])

            max_area_idx = np.argmax(mean_areas)
            min_area_idx = np.argmin(mean_areas)

            f.write(f"3. **Area Media**: {labels[max_area_idx]} ha l'area media maggiore ({mean_areas[max_area_idx]:,.2f} {unit})\n")

            # Determina se sono probabilmente campioni diversi o stesso campione
            area_ratio = mean_areas[max_area_idx] / mean_areas[min_area_idx]
            n_ratio = n_fibers[max_fibers_idx] / n_fibers[min_fibers_idx]

            if area_ratio > 2.0 or n_ratio > 1.5:
                f.write(f"\n⚠️ **Nota**: Le differenze significative in area media (rapporto {area_ratio:.1f}x) e/o numero fibre (rapporto {n_ratio:.1f}x) suggeriscono che potrebbero essere **campioni diversi** piuttosto che stesso tessuto con diversa qualità di imaging.\n")
            else:
                f.write(f"\n✅ **Nota**: Le differenze moderate suggeriscono che potrebbe trattarsi dello stesso tessuto con diversa qualità di imaging o processing.\n")

        f.write("\n---\n\n")
        f.write("*Report generato automaticamente da `compare_analyses.py`*\n")

    print(f"  Report salvato: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Confronta risultati di multiple analisi di fibre muscolari',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:

  # Confronta due analisi
  python scripts/compare_analyses.py \\
    --analyses output_old output_new \\
    --labels "Vecchia Immagine" "Nuova Immagine" \\
    --output comparison_report

  # Confronta tre analisi
  python scripts/compare_analyses.py \\
    --analyses output_sample1 output_sample2 output_sample3 \\
    --labels "Campione 1" "Campione 2" "Campione 3" \\
    --output comparison_3samples
        """
    )

    parser.add_argument('--analyses', nargs='+', required=True,
                       help='Lista di directory output da confrontare (es. output_old output_new)')

    parser.add_argument('--labels', nargs='+', required=True,
                       help='Label per ogni analisi (stesso ordine di --analyses)')

    parser.add_argument('--output', type=str, required=True,
                       help='Directory di output per report e visualizzazioni')

    args = parser.parse_args()

    # Valida input
    if len(args.analyses) != len(args.labels):
        print("❌ Errore: Il numero di --analyses deve corrispondere al numero di --labels")
        sys.exit(1)

    if len(args.analyses) < 2:
        print("❌ Errore: Servono almeno 2 analisi da confrontare")
        sys.exit(1)

    print("="*80)
    print("CONFRONTO ANALISI FIBRE MUSCOLARI")
    print("="*80)

    # Carica tutte le analisi
    analyses = []
    print("\nCaricamento analisi...")
    for output_dir, label in zip(args.analyses, args.labels):
        print(f"  - {label}: {output_dir}")
        try:
            analysis = load_analysis_results(output_dir)
            analyses.append(analysis)
            print(f"    ✓ {analysis['metadata']['n_fibers']:,} fibre")
        except Exception as e:
            print(f"    ✗ Errore: {e}")
            sys.exit(1)

    # Crea output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Genera visualizzazioni
    create_comparison_visualizations(analyses, args.labels, args.output)

    # Genera report
    report_file = output_path / 'COMPARISON_REPORT.md'
    generate_comparison_report(analyses, args.labels, report_file)

    # Salva anche CSV comparativo
    print("\nCreazione CSV comparativo...")
    comparison_data = []
    for analysis, label in zip(analyses, args.labels):
        metadata = analysis['metadata']
        summary = analysis['summary']

        row = {
            'analysis': label,
            'n_fibers': metadata['n_fibers'],
            'image_width_px': metadata['image_dimensions']['width'],
            'image_height_px': metadata['image_dimensions']['height'],
            'mask_coverage_percent': metadata['mask_coverage_percent']
        }

        # Aggiungi tutte le metriche dal summary
        for _, metric_row in summary.iterrows():
            metric_name = metric_row['metric']
            metric_value = metric_row['value']
            row[metric_name] = metric_value

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    csv_file = output_path / 'comparison_summary.csv'
    comparison_df.to_csv(csv_file, index=False)
    print(f"  Salvato: {csv_file}")

    print("\n" + "="*80)
    print("✅ CONFRONTO COMPLETATO")
    print("="*80)
    print(f"\nOutput salvato in: {args.output}/")
    print(f"  - COMPARISON_REPORT.md")
    print(f"  - comparison_summary.csv")
    print(f"  - comparison_area_distributions.png")
    print(f"  - comparison_boxplot.png")
    print(f"  - comparison_metrics.png")
    print()

if __name__ == '__main__':
    main()
