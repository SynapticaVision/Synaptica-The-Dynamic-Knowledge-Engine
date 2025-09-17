import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# The main class, now the Synaptica engine
class Synaptica:
    def __init__(self, max_level=5):
        self.graph = nx.DiGraph()
        self.counter = 0
        self.max_level = max_level
        self.history = {}  # The system's knowledge memory

    # Function to create an idea. It can be random or based on real data.
    def create_idea(self, level=0, real_data_optional=None):
        idea = f"Idea #{self.counter} ✨ Level {level}"

        if real_data_optional:
            # If it comes from an "API", use the provided data
            idea_vector = real_data_optional['attribute_vector']
            idea_name = real_data_optional.get('name', idea)
        else:
            # Otherwise, create a random attribute vector for the simulation
            idea_vector = np.array([random.uniform(0.5, 1.5), random.random(), random.random(), random.random()])
            idea_name = idea

        # Future projection based on the attribute vector
        future_projection = np.sum(idea_vector) * (1 + random.random() * 0.2)

        self.graph.add_node(
            idea_name,
            energy=idea_vector[0],
            magic=idea_vector[1],
            art=idea_vector[2],
            science=idea_vector[3],
            level=level,
            future_projection=future_projection,
            summary=f"E:{idea_vector[0]:.2f} M:{idea_vector[1]:.2f} A:{idea_vector[2]:.2f} S:{idea_vector[3]:.2f} P:{future_projection:.2f}"
        )

        self.history[idea_name] = {
            'energy': idea_vector[0],
            'magic': idea_vector[1],
            'art': idea_vector[2],
            'science': idea_vector[3],
            'impact': 0,
            'future_projection': future_projection
        }
        self.counter += 1

        # Connects the new idea to the 3 most similar existing ones
        if len(self.graph.nodes) > 1:
            others = list(self.graph.nodes)
            others.remove(idea_name)
            similars = sorted(others, key=lambda n: self.similarity(idea_name, n), reverse=True)
            for c in similars[:3]:
                weight = self.similarity(idea_name, c)
                self.graph.add_edge(c, idea_name, weight=weight)

        # Continues the creation of sub-ideas (branches)
        if level < self.max_level:
            for _ in range(random.randint(1, 3)):
                inspiration = self.history.get(random.choice(list(self.history.keys())), None) if self.history else None
                # Sub-ideas are inspired by history, not external data
                sub_idea = self.create_idea(level=level + 1)
                weight = self.similarity(idea_name, sub_idea)
                self.graph.add_edge(idea_name, sub_idea, weight=weight)

        return idea_name

    # Simulation of the data input API
    def add_idea_via_api(self, idea_name, attribute_vector):
        """Simulates the input of a high-impact idea into the platform."""
        data = {
            'name': idea_name,
            'attribute_vector': attribute_vector
        }
        self.create_idea(level=0, real_data_optional=data)

    def evolve_idea(self, idea):
        node = self.graph.nodes[idea]
        for attr in ['magic', 'art', 'science']:
            node[attr] *= 0.9 + 0.2 * random.random()
            self.history[idea][attr] = node[attr]

        node['energy'] *= 0.95 + 0.1 * random.random()
        node['future_projection'] = node['energy'] + node['magic'] + node['art'] + node['science']
        self.history[idea]['future_projection'] = node['future_projection']
        node['summary'] = f"E:{node['energy']:.2f} M:{node['magic']:.2f} A:{node['art']:.2f} S:{node['science']:.2f} P:{node['future_projection']:.2f}"
        self.history[idea]['impact'] += node['future_projection']

    def similarity(self, a, b):
        na = self.graph.nodes[a]
        nb = self.graph.nodes[b]
        va = np.array([na['magic'], na['art'], na['science']])
        vb = np.array([nb['magic'], nb['art'], nb['science']])
        return 1 - np.linalg.norm(va - vb)

    def futuristic_self_correction(self):
        # Removal of low-impact ideas
        for n in list(self.graph.nodes):
            node = self.graph.nodes[n]
            if node['energy'] < 0.3 or node['future_projection'] < 0.5:
                self.graph.remove_node(n)
                self.history.pop(n, None)

        # Consolidation of knowledge: fusion of highly similar ideas
        nodes = list(self.graph.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if self.similarity(a, b) > 0.96:
                    self.fuse_ideas(a, b)

        # Updates edge weights
        for u, v in list(self.graph.edges):
            if u in self.graph.nodes and v in self.graph.nodes:
                self.graph[u][v]['weight'] = self.similarity(u, v)

    def fuse_ideas(self, a, b):
        node_a = self.graph.nodes[a]
        node_b = self.graph.nodes[b]
        energy = (node_a['energy'] + node_b['energy']) / 2
        magic = (node_a['magic'] + node_b['magic']) / 2
        art = (node_a['art'] + node_b['art']) / 2
        science = (node_a['science'] + node_b['science']) / 2
        future_projection = (node_a['future_projection'] + node_b['future_projection']) / 2
        level = max(node_a['level'], node_b['level'])
        new_idea = f"Future Consolidation #{self.counter}"
        self.graph.add_node(
            new_idea,
            energy=energy,
            magic=magic,
            art=art,
            science=science,
            future_projection=future_projection,
            level=level,
            summary=f"E:{energy:.2f} M:{magic:.2f} A:{art:.2f} S:{science:.2f} P:{future_projection:.2f}"
        )
        self.history[new_idea] = {
            'energy': energy, 'magic': magic, 'art': art,
            'science': science, 'impact': self.history[a]['impact'] + self.history[b]['impact'],
            'future_projection': future_projection
        }
        for n in [a, b]:
            for neigh in list(self.graph.successors(n)):
                self.graph.add_edge(new_idea, neigh, weight=self.similarity(new_idea, neigh))
            for neigh in list(self.graph.predecessors(n)):
                self.graph.add_edge(neigh, new_idea, weight=self.similarity(neigh, new_idea))
        self.graph.remove_node(a)
        self.graph.remove_node(b)
        self.history.pop(a, None)
        self.history.pop(b, None)
        self.counter += 1

    def update_frame(self, frame):
        # Every 10 frames, a high-impact idea is injected
        if frame % 10 == 0 and frame > 0:
            print(f"Frame {frame}: Injecting a new high-impact idea!")
            ai_revolution_idea = np.array([2.5, 0.9, 0.2, 0.8])  # An idea with high energy and focus on science
            self.add_idea_via_api("AI Revolution", ai_revolution_idea)

        # Adds some random ideas to simulate "noise" in the system
        for _ in range(random.randint(2, 5)):
            self.create_idea()

        for n in list(self.graph.nodes):
            self.evolve_idea(n)

        self.futuristic_self_correction()

        plt.cla()
        pos = nx.spring_layout(self.graph, seed=None, k=0.5, iterations=50)
        colors = [self.graph.nodes[n]['future_projection'] for n in self.graph.nodes]
        sizes = [300 + 400 * (self.graph.nodes[n]['energy'] + self.graph.nodes[n]['magic']) for n in self.graph.nodes]
        
        nx.draw(
            self.graph, pos,
            with_labels=True,
            labels={n: n for n in self.graph.nodes},
            node_color=colors,
            cmap=plt.cm.plasma,
            node_size=sizes,
            font_size=7,
            alpha=0.9
        )

        for u, v, d in self.graph.edges(data=True):
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=d['weight'] * 0.8)

        plt.title(f"Synaptica: The Digital Collective Brain ✨ Frame: {frame}")

    def animate_multiverse(self, frames=50, interval=800):
        fig = plt.figure(figsize=(16, 12))
        ani = animation.FuncAnimation(fig, self.update_frame, frames=frames, interval=interval)
        plt.show()

# --- Synaptica Execution ---
synaptica_engine = Synaptica()
synaptica_engine.animate_multiverse(frames=50, interval=800)
