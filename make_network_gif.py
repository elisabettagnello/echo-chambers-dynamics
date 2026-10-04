"""
Animate the network snapshots of a run (default: the Fig. 3 run) as a GIF.

The layout of each frame starts from the previous one and only takes a few
spring-layout iterations, so nodes drift smoothly as clusters separate.

Usage:
    python make_network_gif.py
    python make_network_gif.py --input outputs_quick/fig3 --fps 5
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx

NODE_SIZE = 100
EDGE_ALPHA = 0.15
LAYOUT_ITERATIONS = 5  # physics per frame: 3-5 keeps the motion smooth


def make_gif(run_folder, output, fps=5, dpi=150, frame_step=1):
    files = sorted(glob.glob(os.path.join(run_folder, 'network_data', '*.gexf*')))[::frame_step]
    if not files:
        raise SystemExit(f"No network snapshots found in {run_folder}/network_data")
    print(f"Rendering {len(files)} frames...")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.95)
    pos = nx.spring_layout(nx.read_gexf(files[0]), seed=42, k=0.15)

    def update(i):
        nonlocal pos
        ax.clear()
        G = nx.read_gexf(files[i])
        pos = nx.spring_layout(G, pos=pos, k=0.15, iterations=LAYOUT_ITERATIONS, seed=42)
        opinions = [float(G.nodes[n].get('color', 0)) for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, width=0.8, alpha=EDGE_ALPHA, edge_color='#7f7f7f')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=NODE_SIZE, node_color=opinions,
                               cmap='viridis', vmin=-1, vmax=1, edgecolors='white', linewidths=1.5)
        step = int(os.path.basename(files[i]).split('_')[1].split('.')[0])
        ax.set_title(f"Network Evolution • t={step}", fontsize=18, fontweight='light', color='#333333')
        ax.axis('off')
        if i % 5 == 0:
            print(f"    {int(100 * i / len(files))}%")

    ani = animation.FuncAnimation(fig, update, frames=len(files), interval=1000 / fps)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    ani.save(output, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='outputs/fig3', help='run folder containing network_data/')
    parser.add_argument('--output', default='figures/movie_hd_network_evolution.gif')
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--frame-step', type=int, default=1, help='use every n-th snapshot')
    args = parser.parse_args()
    make_gif(args.input, args.output, args.fps, args.dpi, args.frame_step)
