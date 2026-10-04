"""
Turn the simulation outputs into the figures used in the presentation.

Usage:
    python plot_figures.py fig3 fig4           # selected figures
    python plot_figures.py all
    python plot_figures.py --quick all         # read outputs_quick/
    python plot_figures.py all --show          # also open each figure

Figures are saved as figures/figure_<n>.png (the names used by the slides).
"""

import argparse
import glob
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm

COLOR_GREY = '#7f7f7f'
STRATEGIES = ['Random', 'Repost', 'Recommendation']


# ---------------------------------------------------------------------------
# Style and helpers
# ---------------------------------------------------------------------------
def set_conference_style():
    """Large fonts and thick lines, readable when projected."""
    sns.set_context("talk", font_scale=1.2)
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.titleweight': 'bold',
        'lines.linewidth': 3.0,
        'lines.markersize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_csv(folder, filename):
    files = sorted(glob.glob(os.path.join(folder, '**', filename), recursive=True))
    if not files:
        print(f"    missing {filename} in {folder}")
        return None
    return pd.read_csv(files[0], compression='gzip', index_col=0)


def gexf_files(folder):
    return sorted(glob.glob(os.path.join(folder, 'network_data', '*.gexf*')))


def gexf_step(path):
    """G_0000500.gexf.bz2 -> 500"""
    return int(os.path.basename(path).split('_')[1].split('.')[0])


def node_opinions(G):
    return [float(G.nodes[n].get('color', 0)) for n in G.nodes()]


def draw_network(ax, G, title="", k=0.15):
    pos = nx.spring_layout(G, seed=42, k=k, iterations=50)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color=node_opinions(G),
                           cmap='viridis', vmin=-1, vmax=1, edgecolors='white', linewidths=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.8, edge_color='#555')
    ax.set_title(title, pad=10, fontsize=14)
    ax.axis('off')


def draw_final_network(ax, folder, title=""):
    files = gexf_files(folder)
    if not files:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.axis('off')
        return
    draw_network(ax, nx.read_gexf(files[-1]), title)


def plot_trajectories(ax, df_op, linewidth=1.5, alpha=0.6):
    n = df_op.shape[1]
    ax.set_prop_cycle('color', [cm.viridis(i / n) for i in range(n)])
    ax.plot(df_op.index, df_op, linewidth=linewidth, alpha=alpha)
    ax.set_ylim(-1.1, 1.1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig3(out):
    """Screen entropy, opinion trajectories and three network snapshots."""
    path = f'{out}/fig3'
    df_op = load_csv(path, 'opinions.csv.gz')
    df_div = load_csv(path, 'screen_diversity.csv.gz')
    files = gexf_files(path)
    if df_op is None or df_div is None or not files:
        return None

    # Snapshots closest to the target times.
    time_map = {gexf_step(f): f for f in files}
    snaps = [min(time_map, key=lambda t: abs(t - target)) for target in (0, 500, 5000)]
    titles = [f"Start (t={snaps[0]})", f"Mid (t={snaps[1]})", f"End (t={snaps[2]})"]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 0.8], hspace=0.35, wspace=0.25)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    gs_nets = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :], wspace=0.1)

    mean_div, std_div = df_div.mean(axis=1), df_div.std(axis=1)
    ax1.plot(mean_div.index, mean_div, color=cm.viridis(0.1))
    ax1.fill_between(mean_div.index, mean_div - std_div, mean_div + std_div, color=cm.viridis(0.1), alpha=0.2)
    ax1.set_ylabel(r'Screen Entropy $H(S_i)$')
    ax1.set_xlabel('Time (steps)')
    ax1.set_title('Information Diversity', loc='left')
    ax1.set_xlim(left=0)

    plot_trajectories(ax2, df_op)
    ax2.set_ylabel(r'Opinion State ($x_i$)')
    ax2.set_xlabel('Time (steps)')
    ax2.set_title('Opinion Polarization', loc='left')
    ax2.set_xlim(left=0)

    for ax in (ax1, ax2):
        for t in snaps:
            ax.axvline(x=t, color=COLOR_GREY, linestyle='--', alpha=0.5, linewidth=1.5)

    for i, (t, title) in enumerate(zip(snaps, titles)):
        draw_network(fig.add_subplot(gs_nets[i]), nx.read_gexf(time_map[t]), title, k=0.2)

    return fig


def count_clusters(opinions, epsilon):
    """Opinion clusters: a gap larger than epsilon between sorted opinions
    breaks the chain of influence and starts a new cluster."""
    if len(opinions) == 0:
        return 0
    return int(np.sum(np.diff(np.sort(opinions)) > epsilon) + 1)


def fig4(out):
    """Number of clusters and polarization width as a function of epsilon."""
    data = {}
    for folder in glob.glob(f'{out}/fig4/eps_*'):
        eps = float(os.path.basename(folder).split('_')[1])
        df = load_csv(folder, 'opinions.csv.gz')
        if df is None:
            continue
        final = df.iloc[-1].values
        d = data.setdefault(eps, {'clusters': [], 'width': []})
        d['clusters'].append(count_clusters(final, eps))
        d['width'].append(final.max() - final.min())
    if not data:
        return None

    eps = sorted(data)
    stat = lambda key, f: [f(data[e][key]) for e in eps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.errorbar(eps, stat('clusters', np.mean), yerr=stat('clusters', np.std), fmt='-o',
                 color=cm.viridis(0.2), ecolor=COLOR_GREY, capsize=4)
    ax1.set_xlabel(r'Bounded Confidence ($\epsilon$)')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Social Fragmentation', loc='left')
    ax1.set_ylim(bottom=0)

    ax2.errorbar(eps, stat('width', np.mean), yerr=stat('width', np.std), fmt='-s',
                 color=cm.viridis(0.6), ecolor=COLOR_GREY, capsize=4)
    ax2.set_xlabel(r'Bounded Confidence ($\epsilon$)')
    ax2.set_ylabel(r'Max Distance ($O_{max} - O_{min}$)')
    ax2.set_title('Polarization Width', loc='left')
    ax2.set_ylim(-0.1, 2.1)
    fig.tight_layout()
    return fig


def fig5(out):
    """Opinion trajectories and final network for the three regimes."""
    scenarios = [('influence_only', 'Influence Only'),
                 ('rewiring_only', 'Rewiring Only'),
                 ('both', 'Echo Chamber (Full)')]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, (name, title) in enumerate(scenarios):
        path = f'{out}/fig5/{name}'
        df = load_csv(path, 'opinions.csv.gz')
        if df is not None:
            plot_trajectories(axes[0, i], df, alpha=0.5)
        axes[0, i].set_title(title, fontsize=19)
        axes[0, i].set_xlabel('Time')
        if i == 0:
            axes[0, i].set_ylabel('Opinion')
        draw_final_network(axes[1, i], path)
    fig.tight_layout()
    return fig


def fig6(out):
    """Heatmap of the convergence time on the (mu, q) grid."""
    rows = []
    for folder in glob.glob(f'{out}/fig6/mu_*'):
        parts = os.path.basename(folder).split('_')
        df = load_csv(folder, 'opinions.csv.gz')
        if df is not None:
            rows.append({'mu': float(parts[1]), 'q': float(parts[3]), 'time': len(df)})
    if not rows:
        return None
    pivot = pd.DataFrame(rows).pivot(index='mu', columns='q', values='time')

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap='viridis', ax=ax,
                cbar_kws={'label': 'Convergence Time'}, linewidths=1, linecolor='white')
    ax.invert_yaxis()
    ax.set_title('Convergence Time Landscape')
    ax.set_ylabel(r'Influence Strength ($\mu$)')
    ax.set_xlabel('Rewiring Probability (q)')
    fig.tight_layout()
    return fig


def fig7(out):
    """Closed triads (N=100) and in-degree CCDF (large network) per strategy."""
    colors = {'Random': cm.viridis(0.2), 'Repost': cm.viridis(0.5), 'Recommendation': cm.viridis(0.85)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    means, stds = [], []
    for strat in STRATEGIES:
        counts = []
        for run_folder in glob.glob(f'{out}/fig7a_triads/{strat}/run_*'):
            files = gexf_files(run_folder)
            if files:
                G = nx.to_undirected(nx.read_gexf(files[-1]))
                counts.append(sum(nx.triangles(G).values()) / 3)
        means.append(np.mean(counts) if counts else 0)
        stds.append(np.std(counts) if counts else 0)

    ax1.bar(STRATEGIES, means, yerr=stds, capsize=6, color=[colors[s] for s in STRATEGIES],
            alpha=0.9, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Clustering (Closed Triads)')
    ax1.set_title('Local Structure', loc='left')
    ax1.grid(axis='x', alpha=0)

    for strat in STRATEGIES:
        files = gexf_files(f'{out}/fig7b_degrees/{strat}')
        if not files:
            continue
        degrees = sorted((d for _, d in nx.read_gexf(files[-1]).in_degree()), reverse=True)
        ccdf = np.arange(1, len(degrees) + 1) / len(degrees)
        ax2.loglog(degrees, ccdf, label=strat, color=colors[strat], linestyle='--', alpha=0.9)

    ax2.set_xlabel(r'In-degree ($k_{in}$)')
    ax2.set_ylabel(r'CCDF $P(K \geq k_{in})$')
    ax2.set_title('Follower Distribution', loc='left')
    ax2.legend(frameon=True, framealpha=1, fancybox=False, edgecolor='white')
    fig.tight_layout()
    return fig


def fig10(out):
    """Kernel density of opinions at the start and at the end of a large run."""
    df = load_csv(f'{out}/fig10', 'opinions.csv.gz')
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(11, 7))
    for row, color, label, alpha in [(0, cm.viridis(0.9), 'Initial State', 0.4),
                                     (-1, cm.viridis(0.1), 'Final State', 0.6)]:
        sns.kdeplot(df.iloc[row].values, fill=True, color=color, alpha=alpha,
                    label=label, edgecolor=color, ax=ax)
    ax.set_title('Evolution of Opinion Density')
    ax.set_xlabel('Opinion Space ($x$)')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(-1.1, 1.1)
    ax.legend(loc='upper left', borderaxespad=0., frameon=True, edgecolor='white')
    fig.tight_layout()
    return fig


FIGURES = {'fig3': (fig3, 'figure_3.png'), 'fig4': (fig4, 'figure_4.png'),
           'fig5': (fig5, 'figure_5.png'), 'fig6': (fig6, 'figure_6.png'),
           'fig7': (fig7, 'figure_7.png'), 'fig10': (fig10, 'figure_10.png')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('figures', nargs='+', choices=list(FIGURES) + ['all'])
    parser.add_argument('--quick', action='store_true', help='read outputs_quick/ instead of outputs/')
    parser.add_argument('--out', default=None, help='simulation output folder to read')
    parser.add_argument('--figdir', default='figures', help='where to save the PNGs')
    parser.add_argument('--show', action='store_true', help='also display each figure')
    args = parser.parse_args()

    if not args.show:
        matplotlib.use('Agg')
    set_conference_style()
    out = args.out or ('outputs_quick' if args.quick else 'outputs')
    os.makedirs(args.figdir, exist_ok=True)

    for name in (list(FIGURES) if 'all' in args.figures else args.figures):
        func, filename = FIGURES[name]
        print(f"[{name}]")
        fig = func(out)
        if fig is None:
            print("    no data found, skipped")
            continue
        fig.savefig(os.path.join(args.figdir, filename), dpi=150, bbox_inches='tight')
        print(f"    saved {args.figdir}/{filename}")
        if args.show:
            plt.show()
        plt.close(fig)
