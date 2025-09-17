import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# A classe principal, agora o motor da Synaptica
class Synaptica:
    def __init__(self, max_nivel=5):
        self.grafo = nx.DiGraph()
        self.contador = 0
        self.max_nivel = max_nivel
        self.historico = {}  # Memória de conhecimento do sistema

    # Função para criar uma ideia no motor. Pode ser aleatória ou baseada em dados reais.
    def criar_ideia(self, nivel=0, dados_reais_opcionais=None):
        ideia = f"Ideia #{self.contador} ✨ Nível {nivel}"

        if dados_reais_opcionais:
            # Se vier de uma "API", use os dados fornecidos
            vetor_ideia = dados_reais_opcionais['vetor_de_atributos']
            nome_ideia = dados_reais_opcionais.get('nome', ideia)
        else:
            # Caso contrário, crie um vetor de atributos aleatório para a simulação
            vetor_ideia = np.array([random.uniform(0.5, 1.5), random.random(), random.random(), random.random()])
            nome_ideia = ideia

        # Projeção futura baseada no vetor de atributos
        proj_futuro = np.sum(vetor_ideia) * (1 + random.random() * 0.2)

        self.grafo.add_node(
            nome_ideia,
            energia=vetor_ideia[0],
            magia=vetor_ideia[1],
            arte=vetor_ideia[2],
            ciencia=vetor_ideia[3],
            nivel=nivel,
            proj_futuro=proj_futuro,
            resumo=f"E:{vetor_ideia[0]:.2f} M:{vetor_ideia[1]:.2f} A:{vetor_ideia[2]:.2f} C:{vetor_ideia[3]:.2f} P:{proj_futuro:.2f}"
        )

        self.historico[nome_ideia] = {
            'energia': vetor_ideia[0],
            'magia': vetor_ideia[1],
            'arte': vetor_ideia[2],
            'ciencia': vetor_ideia[3],
            'impacto': 0,
            'proj_futuro': proj_futuro
        }
        self.contador += 1

        # Conecta a nova ideia às 3 mais similares existentes
        if len(self.grafo.nodes) > 1:
            outras = list(self.grafo.nodes)
            outras.remove(nome_ideia)
            similares = sorted(outras, key=lambda n: self.similaridade(nome_ideia, n), reverse=True)
            for c in similares[:3]:
                peso = self.similaridade(nome_ideia, c)
                self.grafo.add_edge(c, nome_ideia, weight=peso)

        # Continua a criação de sub-ideias (ramificações)
        if nivel < self.max_nivel:
            for _ in range(random.randint(1, 3)):
                inspiracao_sub = self.historico.get(random.choice(list(self.historico.keys())), None) if self.historico else None
                # As sub-ideias são inspiradas pelo histórico, não por dados externos
                sub = self.criar_ideia(nivel=nivel + 1)
                peso = self.similaridade(nome_ideia, sub)
                self.grafo.add_edge(nome_ideia, sub, weight=peso)

        return nome_ideia

    # Simulação da API de entrada de dados
    def adicionar_ideia_via_api(self, nome_da_ideia, vetor_de_atributos):
        """Simula a entrada de uma ideia de alto impacto na plataforma."""
        dados = {
            'nome': nome_da_ideia,
            'vetor_de_atributos': vetor_de_atributos
        }
        self.criar_ideia(nivel=0, dados_reais_opcionais=dados)

    def evoluir_ideia(self, ideia):
        node = self.grafo.nodes[ideia]
        for attr in ['magia', 'arte', 'ciencia']:
            node[attr] *= 0.9 + 0.2 * random.random()
            self.historico[ideia][attr] = node[attr]

        node['energia'] *= 0.95 + 0.1 * random.random()
        node['proj_futuro'] = node['energia'] + node['magia'] + node['arte'] + node['ciencia']
        self.historico[ideia]['proj_futuro'] = node['proj_futuro']
        node['resumo'] = f"E:{node['energia']:.2f} M:{node['magia']:.2f} A:{node['arte']:.2f} C:{node['ciencia']:.2f} P:{node['proj_futuro']:.2f}"
        self.historico[ideia]['impacto'] += node['proj_futuro']

    def similaridade(self, a, b):
        na = self.grafo.nodes[a]
        nb = self.grafo.nodes[b]
        va = np.array([na['magia'], na['arte'], na['ciencia']])
        vb = np.array([nb['magia'], nb['arte'], nb['ciencia']])
        return 1 - np.linalg.norm(va - vb)

    def autocorrecao_futurista(self):
        # Remoção de ideias de baixo impacto
        for n in list(self.grafo.nodes):
            node = self.grafo.nodes[n]
            if node['energia'] < 0.3 or node['proj_futuro'] < 0.5:
                self.grafo.remove_node(n)
                self.historico.pop(n, None)

        # Consolidação do conhecimento: fusão de ideias altamente similares
        nodes = list(self.grafo.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if self.similaridade(a, b) > 0.96:
                    self.fundir_ideias(a, b)

        # Atualiza pesos das arestas
        for u, v in list(self.grafo.edges):
            if u in self.grafo.nodes and v in self.grafo.nodes:
                self.grafo[u][v]['weight'] = self.similaridade(u, v)

    def fundir_ideias(self, a, b):
        node_a = self.grafo.nodes[a]
        node_b = self.grafo.nodes[b]
        energia = (node_a['energia'] + node_b['energia']) / 2
        magia = (node_a['magia'] + node_b['magia']) / 2
        arte = (node_a['arte'] + node_b['arte']) / 2
        ciencia = (node_a['ciencia'] + node_b['ciencia']) / 2
        proj_futuro = (node_a['proj_futuro'] + node_b['proj_futuro']) / 2
        nivel = max(node_a['nivel'], node_b['nivel'])
        nova = f"Consolidação Futura #{self.contador}"
        self.grafo.add_node(
            nova,
            energia=energia,
            magia=magia,
            arte=arte,
            ciencia=ciencia,
            proj_futuro=proj_futuro,
            nivel=nivel,
            resumo=f"E:{energia:.2f} M:{magia:.2f} A:{arte:.2f} C:{ciencia:.2f} P:{proj_futuro:.2f}"
        )
        self.historico[nova] = {
            'energia': energia, 'magia': magia, 'arte': arte,
            'ciencia': ciencia, 'impacto': self.historico[a]['impacto'] + self.historico[b]['impacto'],
            'proj_futuro': proj_futuro
        }
        for n in [a, b]:
            for neigh in list(self.grafo.successors(n)):
                self.grafo.add_edge(nova, neigh, weight=self.similaridade(nova, neigh))
            for neigh in list(self.grafo.predecessors(n)):
                self.grafo.add_edge(neigh, nova, weight=self.similaridade(neigh, nova))
        self.grafo.remove_node(a)
        self.grafo.remove_node(b)
        self.historico.pop(a, None)
        self.historico.pop(b, None)
        self.contador += 1

    def atualizar_frame(self, frame):
        # A cada 10 frames, uma ideia de alto impacto é injetada
        if frame % 10 == 0 and frame > 0:
            print(f"Frame {frame}: Injetando nova ideia de alto impacto!")
            nova_ideia_IA = np.array([2.5, 0.9, 0.2, 0.8])  # Uma ideia com alta energia e foco em ciência
            self.adicionar_ideia_via_api("Revolução da IA", nova_ideia_IA)

        # Adiciona algumas ideias aleatórias para simular o "ruído" do sistema
        for _ in range(random.randint(2, 5)):
            self.criar_ideia()

        for n in list(self.grafo.nodes):
            self.evoluir_ideia(n)

        self.autocorrecao_futurista()

        plt.cla()
        pos = nx.spring_layout(self.grafo, seed=None, k=0.5, iterations=50)
        cores = [self.grafo.nodes[n]['proj_futuro'] for n in self.grafo.nodes]
        tamanhos = [300 + 400 * (self.grafo.nodes[n]['energia'] + self.grafo.nodes[n]['magia']) for n in self.grafo.nodes]
        
        nx.draw(
            self.grafo, pos,
            with_labels=True,
            labels={n: n for n in self.grafo.nodes},
            node_color=cores,
            cmap=plt.cm.plasma,
            node_size=tamanhos,
            font_size=7,
            alpha=0.9
        )

        for u, v, d in self.grafo.edges(data=True):
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=d['weight'] * 0.8)

        plt.title(f"Synaptica: O Cérebro Coletivo Digital ✨ Frame: {frame}")

    def animar_multiverso(self, frames=50, interval=800):
        fig = plt.figure(figsize=(16, 12))
        ani = animation.FuncAnimation(fig, self.atualizar_frame, frames=frames, interval=interval)
        plt.show()

# --- Execução da Synaptica ---
synaptica_engine = Synaptica()
synaptica_engine.animar_multiverso(frames=50, interval=800)
