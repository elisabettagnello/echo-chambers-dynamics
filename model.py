"""
Agent-based model of echo-chamber formation on a directed social network.

Reproduction of Sasahara et al. (2021), "Social influence and unfollowing
accelerate the emergence of echo chambers", J. Comput. Soc. Sci. 4, 381-402.

Each agent holds a continuous opinion in [-1, 1] and reads a limited "screen":
the L most recent messages posted by the accounts it follows. At every step a
randomly chosen agent:

  1. reads its screen and splits the messages into concordant and discordant
     ones (bounded confidence: |opinion - message| < epsilon);
  2. moves its opinion towards the mean of the concordant messages (strength mu);
  3. with probability q, unfollows the author of a discordant message and
     follows someone else (Random, Repost or Recommendation strategy);
  4. reposts a concordant message (probability p) or posts its own opinion.

A run stops when the network has split into two or more weakly connected
components, each internally converged (opinion range <= epsilon).
"""

import os
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats

MESSAGE_COLUMNS = ['msg_id', 'orig_msg_id', 'who_posted', 'who_originated', 'content']


def screen_diversity(content_values, bins):
    """Shannon entropy (in bits) of the opinion histogram of a screen."""
    if len(content_values) == 0:
        return 0.0
    h, _ = np.histogram(content_values, range=(-1, 1), bins=bins)
    return stats.entropy(h, base=2)


class Message:
    # __slots__ keeps memory low: the message buffer can hold ~10^5 objects.
    __slots__ = MESSAGE_COLUMNS

    def __init__(self, msg_id, orig_msg_id, who_posted, who_originated, content):
        self.msg_id = msg_id
        self.orig_msg_id = orig_msg_id
        self.who_posted = who_posted
        self.who_originated = who_originated
        self.content = content

    def to_dict(self):
        return {c: getattr(self, c) for c in MESSAGE_COLUMNS}


class SocialMedia:
    """Directed follower graph plus a bounded, time-ordered message buffer."""

    def __init__(self, num_agents, num_links, l, sns_seed):
        self.num_agents = num_agents
        random_state = np.random.RandomState(sns_seed)
        self.G = nx.gnm_random_graph(n=num_agents, m=num_links, seed=random_state, directed=True)
        self.modify_random_graph()

        # System memory: num_agents * 50 messages is enough for every agent to
        # find recent posts from the accounts it follows.
        self.buffer_size = num_agents * 50
        # deque(maxlen=...) drops the oldest message automatically.
        self.message_history = deque(maxlen=self.buffer_size)
        self.screen_size = l

    def modify_random_graph(self):
        """Ensure every node follows at least one account (out-degree > 0)."""
        for no_outdegree_node in [k for k, v in list(self.G.out_degree()) if v == 0]:
            candidates = [k for k, v in list(self.G.out_degree()) if v >= 2]
            if not candidates:
                continue
            target_node = np.random.choice(candidates)

            edges = list(self.G.edges(target_node))
            if not edges:
                continue
            target_edge = edges[np.random.choice(len(edges))]

            self.G.remove_edge(target_edge[0], target_edge[1])
            self.G.add_edge(no_outdegree_node, target_edge[1])

    def set_node_colors(self, node_colors):
        for i, c in enumerate(node_colors):
            self.G.nodes[i]['color'] = c

    def show_screen(self, user_id):
        """Return the L most recent messages from followed accounts (FIFO screen).

        Scans the buffer backwards and stops as soon as L messages are found,
        instead of filtering the whole history. Messages originally written by
        the user itself are excluded.
        """
        friends = set(self.G.neighbors(user_id))
        screen_msgs = []
        for msg in reversed(self.message_history):
            if msg.who_posted in friends and msg.who_originated != user_id:
                screen_msgs.append(msg)
                if len(screen_msgs) >= self.screen_size:
                    break
        screen_msgs.reverse()  # back to chronological order

        if not screen_msgs:
            return pd.DataFrame(columns=MESSAGE_COLUMNS)
        return pd.DataFrame([m.to_dict() for m in screen_msgs])

    def update_message_db(self, t, msg):
        self.message_history.append(msg)

    def recommend_similar_users(self, user_id, epsilon, num_agents):
        """Authors of recent messages close to the user's last own message."""
        if not self.message_history:
            return []

        df = pd.DataFrame([m.to_dict() for m in self.message_history])
        similar_users = []

        my_message_df = df[df.who_originated == user_id].tail(1)
        if len(my_message_df) > 0:
            last_message = my_message_df.content.values[0]
            friends = list(self.G.neighbors(user_id))

            recent_global_msgs = df[df.who_originated != user_id].tail(num_agents)
            if not recent_global_msgs.empty:
                similar = recent_global_msgs[abs(last_message - recent_global_msgs.content) < epsilon]
                if len(similar) > 0:
                    similar_users = [u for u in similar.who_originated.values if u not in friends]

        return similar_users

    def rewire_users(self, user_id, unfollow_id, follow_id):
        if self.G.has_edge(user_id, unfollow_id):
            self.G.remove_edge(user_id, unfollow_id)
        if user_id != follow_id and not self.G.has_edge(user_id, follow_id):
            self.G.add_edge(user_id, follow_id)


class Agent:
    def __init__(self, user_id, epsilon, screen_diversity):
        self.user_id = user_id
        self.opinion = np.random.uniform(-1.0, 1.0)
        self.epsilon = epsilon
        self.screen_diversity = screen_diversity
        self.concordant_msgs = []
        self.discordant_msgs = []

    def evaluate_messages(self, screen):
        """Bounded confidence: split the screen into concordant / discordant."""
        self.concordant_msgs = []
        self.discordant_msgs = []
        if len(screen) > 0:
            distance = abs(self.opinion - screen.content)
            self.concordant_msgs = screen[distance < self.epsilon]
            self.discordant_msgs = screen[distance >= self.epsilon]

    def update_opinion(self, mu):
        """Social influence: move towards the mean concordant opinion."""
        if len(self.concordant_msgs) > 0:
            self.opinion = self.opinion + mu * np.mean(self.concordant_msgs.content - self.opinion)
            self.opinion = min(1.0, max(-1.0, self.opinion))

    def post_message(self, msg_id, p):
        if len(self.concordant_msgs) > 0 and np.random.random() < p:
            # Repost a concordant message chosen at random.
            idx = np.random.choice(self.concordant_msgs.index)
            selected = self.concordant_msgs.loc[idx]
            return Message(msg_id=int(msg_id), orig_msg_id=int(selected.orig_msg_id),
                           who_posted=int(self.user_id), who_originated=int(selected.who_originated),
                           content=selected.content)
        # Post a new message expressing the current opinion.
        return Message(msg_id=int(msg_id), orig_msg_id=int(msg_id),
                       who_posted=int(self.user_id), who_originated=int(self.user_id),
                       content=self.opinion)

    def decide_follow_id_at_random(self, friends, num_agents):
        prohibited = set(friends) | {self.user_id}
        candidates = [i for i in range(num_agents) if i not in prohibited]
        if not candidates:
            return self.user_id  # fallback: rewire_users ignores self-follows
        return int(np.random.choice(candidates))

    def decide_unfollow_id_at_random(self, discordant_messages):
        return int(np.random.choice(discordant_messages.who_posted.values))

    def decide_to_rewire(self, social_media, following_methods):
        """Unfollow a discordant author and pick a new account to follow."""
        unfollow_id = None
        follow_id = None
        if len(self.discordant_msgs) == 0:
            return unfollow_id, follow_id

        unfollow_id = self.decide_unfollow_id_at_random(self.discordant_msgs)
        method = np.random.choice(following_methods)
        friends = list(social_media.G.neighbors(self.user_id))
        n = social_media.G.number_of_nodes()

        if method == 'Repost':
            # Original authors of concordant reposts (friends of friends).
            fof = list(set(self.concordant_msgs.who_originated.values) - set(friends))
            follow_id = int(np.random.choice(fof)) if fof else self.decide_follow_id_at_random(friends, n)
        elif method == 'Recommendation':
            similar = social_media.recommend_similar_users(self.user_id, self.epsilon, n)
            follow_id = int(np.random.choice(similar)) if similar else self.decide_follow_id_at_random(friends, n)
        else:  # 'Random'
            follow_id = self.decide_follow_id_at_random(friends, n)

        return unfollow_id, follow_id


class EchoChamberDynamics:
    def __init__(self, num_agents, num_links, epsilon, sns_seed, l, data_dir, gexf_every=500):
        self.num_agents = num_agents
        self.l = l
        self.epsilon = epsilon
        self.set_agents(num_agents, epsilon)
        self.social_media = SocialMedia(num_agents, num_links, l, sns_seed)
        self.data_dir = data_dir
        self.gexf_every = gexf_every  # network snapshot interval (used by Fig. 3 and the GIF)
        self.opinion_data = []
        self.screen_diversity_data = []
        self.global_msg_counter = 0  # guarantees unique message ids

        os.makedirs(os.path.join(data_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'network_data'), exist_ok=True)

    def set_agents(self, num_agents, epsilon):
        initial_diversity = screen_diversity([], bins=10)
        self.agents = [Agent(i, epsilon, initial_diversity) for i in range(num_agents)]

    def is_stationary_state(self, G):
        """True when >= 2 components exist and each has opinion range <= epsilon."""
        components = [G.subgraph(c) for c in nx.weakly_connected_components(G)]
        if len(components) < 2:
            return False
        for C in components:
            opinions = np.array([self.agents[i].opinion for i in C.nodes()])
            if len(opinions) > 0 and np.max(opinions) - np.min(opinions) > self.epsilon:
                return False
        return True

    def export_csv(self, data, ofname):
        path = os.path.join(self.data_dir, 'data', ofname)
        pd.DataFrame(data).to_csv(path, compression='gzip')

    def export_gexf(self, t):
        path = os.path.join(self.data_dir, 'network_data', 'G_' + str(t).zfill(7) + '.gexf.bz2')
        self.social_media.set_node_colors([float(a.opinion) for a in self.agents])
        nx.write_gexf(self.social_media.G, path)

    def final_exports(self, t):
        self.export_csv(self.opinion_data, 'opinions.csv.gz')
        self.export_csv(self.screen_diversity_data, 'screen_diversity.csv.gz')
        df = pd.DataFrame([m.to_dict() for m in self.social_media.message_history])
        df.to_csv(os.path.join(self.data_dir, 'data', 'messages.csv.gz'), compression='gzip')
        self.export_gexf(t)

    def evolve(self, t_max, mu, p, q, rewiring_methods=('Random', 'Repost', 'Recommendation')):
        rewiring_methods = list(rewiring_methods)
        for t in range(t_max):
            if t % 5000 == 0:
                print(f"    step {t}")

            # One row per step: opinions and screen entropy of every agent.
            self.opinion_data.append([a.opinion for a in self.agents])
            self.screen_diversity_data.append([a.screen_diversity for a in self.agents])

            if t % self.gexf_every == 0:
                self.export_gexf(t)

            user_id = np.random.choice(self.num_agents)
            agent = self.agents[user_id]

            # 1. Read the screen.
            screen = self.social_media.show_screen(user_id)
            agent.evaluate_messages(screen)
            agent.screen_diversity = screen_diversity(screen.content.values, bins=10)

            # 2. Social influence.
            agent.update_opinion(mu)

            # 3. Rewiring.
            if np.random.random() < q:
                unfollow_id, follow_id = agent.decide_to_rewire(self.social_media, rewiring_methods)
                if unfollow_id is not None and follow_id is not None:
                    self.social_media.rewire_users(user_id, unfollow_id, follow_id)

            # 4. Post or repost.
            self.global_msg_counter += 1
            msg = agent.post_message(self.global_msg_counter, p)
            self.social_media.update_message_db(t, msg)

            if self.is_stationary_state(self.social_media.G):
                self.final_exports(t)
                print(f"    stationary state reached at t={t}")
                break
            if t >= t_max - 1:
                self.final_exports(t)
                print(f"    t_max reached (t={t}) without a stationary state")
                break
