"""
Generate the simulation data behind each figure of the presentation.

Usage:
    python run_simulations.py fig3 fig4        # selected figures
    python run_simulations.py all              # everything (slow)
    python run_simulations.py --quick all      # tiny runs to test the pipeline

Each run writes to <out>/<figure>/... and is skipped if its data already exist,
so an interrupted batch can be resumed.
"""

import argparse
import os

import numpy as np

from model import EchoChamberDynamics

L = 10          # screen size
SEED = 1        # base seed (run r uses SEED + r)
P_REPOST = 0.5  # repost probability, fixed in every experiment


def run(path, n_agents, n_links, epsilon, mu, q, strategies, seed, t_max):
    """Run one simulation into `path`, unless it has already been done."""
    if os.path.exists(os.path.join(path, 'data', 'opinions.csv.gz')):
        print(f"--> {path}: already done, skipping")
        return
    print(f"--> {path}: N={n_agents}, eps={epsilon}, mu={mu}, q={q}, {strategies}")
    np.random.seed(seed)  # agent opinions and dynamics; the graph has its own seed
    d = EchoChamberDynamics(n_agents, n_links, epsilon, seed, L, path)
    d.evolve(t_max, mu, P_REPOST, q, strategies)


def fig3(out, quick):
    """Single standard run: entropy drop, polarization, segregation over time."""
    run(f'{out}/fig3', 100, 400, epsilon=0.45, mu=0.5, q=0.5,
        strategies=['Random'], seed=SEED, t_max=2000 if quick else 6000)


def fig4(out, quick):
    """Sweep of the bounded-confidence threshold epsilon, 20 runs per value."""
    epsilons = [0.2, 0.6] if quick else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_runs = 2 if quick else 20
    for eps in epsilons:
        for r in range(n_runs):
            run(f'{out}/fig4/eps_{eps:.1f}_run_{r}', 100, 400, epsilon=eps, mu=0.5, q=0.5,
                strategies=['Random'], seed=SEED + r, t_max=1000 if quick else 6000)


def fig5(out, quick):
    """Influence only vs rewiring only vs both (weak mu and q, hence long runs)."""
    scenarios = {
        'influence_only': dict(mu=0.1, q=0.0),
        'rewiring_only': dict(mu=0.0, q=0.1),
        'both': dict(mu=0.1, q=0.1),
    }
    for name, par in scenarios.items():
        run(f'{out}/fig5/{name}', 100, 400, epsilon=0.5, strategies=['Random'],
            seed=SEED, t_max=2000 if quick else 40000, **par)


def fig6(out, quick):
    """Convergence time on a (mu, q) grid, one run per cell."""
    grid = [0.01, 0.1, 1.0] if quick else \
        [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
    for mu in grid:
        for q in grid:
            run(f'{out}/fig6/mu_{mu:.10f}_q_{q:.10f}', 100, 400, epsilon=0.4, mu=mu, q=q,
                strategies=['Random'], seed=SEED, t_max=1000 if quick else 5000)


def fig7a(out, quick):
    """Closed triads for each follow strategy (N=100, 20 runs)."""
    n_runs = 2 if quick else 20
    for strat in ['Random', 'Repost', 'Recommendation']:
        for r in range(n_runs):
            run(f'{out}/fig7a_triads/{strat}/run_{r}', 100, 400, epsilon=0.4, mu=0.5, q=0.5,
                strategies=[strat], seed=SEED + r, t_max=1000 if quick else 20000)


def fig7b(out, quick):
    """In-degree distribution for each follow strategy (one large run each)."""
    n, m = (200, 2000) if quick else (1000, 10000)
    for strat in ['Random', 'Repost', 'Recommendation']:
        run(f'{out}/fig7b_degrees/{strat}', n, m, epsilon=0.5, mu=0.5, q=0.5,
            strategies=[strat], seed=SEED, t_max=1000 if quick else 30000)


def fig10(out, quick):
    """Initial vs final opinion density on a large network."""
    n, m = (300, 1200) if quick else (10000, 40000)
    run(f'{out}/fig10', n, m, epsilon=0.5, mu=0.5, q=0.5,
        strategies=['Random'], seed=SEED, t_max=1000 if quick else 30000)


FIGURES = {'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'fig6': fig6,
           'fig7a': fig7a, 'fig7b': fig7b, 'fig10': fig10}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('figures', nargs='+', choices=list(FIGURES) + ['all'])
    parser.add_argument('--quick', action='store_true', help='tiny runs to check the pipeline end to end')
    parser.add_argument('--out', default=None, help='output folder (default: outputs, or outputs_quick)')
    args = parser.parse_args()

    out = args.out or ('outputs_quick' if args.quick else 'outputs')
    selected = list(FIGURES) if 'all' in args.figures else args.figures
    for name in selected:
        print(f"\n[{name}]")
        FIGURES[name](out, args.quick)
