# Echo chambers from social influence and unfollowing

A Python reproduction of the agent-based model in
**Sasahara et al. (2021), "Social influence and unfollowing accelerate the emergence of echo chambers"**,
*Journal of Computational Social Science* 4, 381–402. 

Project for the *Physics of Complex Systems* course (M.Sc. Physics, Sapienza University of Rome).

## The question

Can echo chambers emerge on a social network without fake news or complex cognitive biases, only from
two simple behaviours: being influenced by like-minded posts, and unfollowing people we disagree with?

## The model

Each agent has an opinion in [-1, 1] and sees only the last *L* messages from the accounts it follows.
At each step a random agent reads its screen, moves its opinion towards the concordant messages
(bounded confidence ε, influence strength μ), with probability *q* unfollows the author of a discordant
message and follows someone else (Random, Repost or Recommendation strategy), and finally posts or reposts.
A run ends when the network has split into internally converged, disconnected communities.

## What is reproduced

- How do information diversity, opinions and network co-evolve in a single run? | `fig3`
- How does tolerance ε control fragmentation vs consensus? (20 runs per ε) | `fig4`
- Is influence alone, or rewiring alone, enough to create echo chambers? | `fig5`
- How fast do echo chambers form across the (μ, q) plane? | `fig6`
- How do follow strategies shape local clustering and the follower distribution? | `fig7a` , `fig7b` , `fig7`
- What does polarization look like in the opinion density of a large network? | `fig10` 

The empirical validation on the US-politics retweet network shown in the presentation uses the
paper's own figure and was not re-run here.

## Running it

```bash
pip install -r requirements.txt

# End-to-end check with tiny runs (about a minute)
python run_simulations.py --quick all
python plot_figures.py --quick all

# Full runs: choose the figures you need; the larger networks (fig7b, fig10) are slow
python run_simulations.py fig3 fig4 fig5
python plot_figures.py fig3 fig4 fig5
python make_network_gif.py            # animation of the Fig. 3 run
```

Simulation data go to `outputs/` (git-ignored), figures to `figures/`. Finished runs are skipped,
so an interrupted batch can be resumed. Each run is seeded, so results are reproducible.

## Repository layout

```
model.py              agents, social network, dynamics and data export
run_simulations.py    parameters of every experiment, one function per figure
plot_figures.py       figures as used in the slides (figures/figure_<n>.png)
make_network_gif.py   animated network evolution
presentation/         slides (HTML)
```

## Notes

- Opinions and screen entropy are stored for every agent at every step, which is convenient for
  plotting but memory-heavy for the 10,000-agent run of Fig. 10.
- The screen is scanned backwards and stops after *L* matching messages, and the message buffer is a
  fixed-size queue; this keeps a step cheap even when the buffer holds hundreds of thousands of messages.

## Reference

Sasahara, K., Chen, W., Peng, H., Ciampaglia, G. L., Flammini, A., & Menczer, F. (2021).
Social influence and unfollowing accelerate the emergence of echo chambers.
*Journal of Computational Social Science*, 4, 381–402.
