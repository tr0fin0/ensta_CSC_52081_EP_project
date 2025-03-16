# Agent_CEM.py
import torch
import numpy as np
import csv
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class Agent_CEM:
    """
    Agente utilizando o método de Cross-Entropy (CEM) para otimização da política.

    Atributos:
        model (torch.nn.Module): Rede neural utilizada como política.
        mean (torch.Tensor): Vetor plano com os parâmetros atuais (média da distribuição).
        sigma (float): Desvio padrão para a amostragem.
        population_size (int): Número de candidatos por geração.
        elite_frac (float): Fração dos melhores candidatos (elite).
        num_elites (int): Número de elites (population_size * elite_frac).
        device (torch.device): Dispositivo para computação.
        dir_models (Path): Diretório para salvar modelos.
        dir_logs (Path): Diretório para salvar logs.

    Métodos:
        get_action(state): Retorna a ação contínua para um dado estado.
        sample_candidate(): Gera um candidato (vetor de parâmetros) com ruído.
        update_policy(candidates, rewards): Atualiza a política usando os candidatos elite.
        save(save_name): Salva o modelo atual e parâmetros da distribuição.
        load(model_name): Carrega modelo e parâmetros salvos.
        write_log(...): Registra métricas de treinamento em arquivo CSV.
    """

    def __init__(self, state_shape, action_dim, device, directory_models, directory_logs, CNN,
                 population_size=50, elite_frac=0.2, initial_std=0.1, load_state=False, load_model=None):
        self.device = device
        self.dir_models = directory_models
        self.dir_logs = directory_logs

        self.population_size = population_size
        self.elite_frac = elite_frac
        self.num_elites = int(self.population_size * self.elite_frac)
        self.sigma = initial_std

        # Inicializa a rede de política (CNN)
        self.model = CNN(state_shape, action_dim).float().to(self.device)

        # Inicializa o vetor de parâmetros (média) a partir do modelo
        self.mean = parameters_to_vector(self.model.parameters()).detach().clone()

        if load_state:
            if load_model is None:
                raise ValueError("Especifique o nome do modelo para carregar.")
            self.load(load_model)

    def get_action(self, state):
        """
        Retorna a ação contínua para o estado dado.

        Para CarRacing-v3 contínuo, o espaço de ação é:
         - steering: [-1, 1]
         - gas: [0, 1]
         - brake: [0, 1]

        A rede gera três valores que são processados (tanh para steering, sigmoid para os demais).
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state_tensor)
        steering = torch.tanh(action[0, 0])
        gas = torch.sigmoid(action[0, 1])
        brake = torch.sigmoid(action[0, 2])
        return np.array([steering.item(), gas.item(), brake.item()])

    def sample_candidate(self):
        """
        Gera um vetor candidato de parâmetros a partir da média atual, adicionando ruído gaussiano.
        """
        noise = torch.randn_like(self.mean) * self.sigma
        candidate = self.mean + noise
        return candidate, noise

    def update_policy(self, candidates, rewards):
        """
        Atualiza a política com base nos candidatos elite.
        """
        rewards = np.array(rewards)
        elite_indices = rewards.argsort()[-self.num_elites:]
        elites = [candidates[i] for i in elite_indices]
        new_mean = torch.stack(elites, dim=0).mean(dim=0)
        # Atualiza sigma como o desvio padrão médio entre os elites
        new_sigma = torch.stack(elites, dim=0).std(dim=0).mean().item()
        self.mean = new_mean
        self.sigma = new_sigma
        # Atualiza os parâmetros do modelo
        vector_to_parameters(self.mean, self.model.parameters())

    def save(self, save_name='CEM'):
        """
        Salva o modelo e os parâmetros da distribuição em arquivo.
        """
        save_path = str(self.dir_models / f"{save_name}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'mean': self.mean,
            'sigma': self.sigma
        }, save_path)
        print(f"Modelo salvo em {save_path}")

    def load(self, model_name):
        """
        Carrega o modelo e os parâmetros da distribuição.
        """
        loaded = torch.load(str(self.dir_models / model_name))
        self.model.load_state_dict(loaded['model_state_dict'])
        self.mean = loaded['mean']
        self.sigma = loaded['sigma']
        vector_to_parameters(self.mean, self.model.parameters())
        print(f"Modelo {model_name} carregado.")

    def write_log(self, generations, best_rewards, avg_rewards, sigmas, log_filename='log_CEM.csv'):
        """
        Escreve os logs de treinamento em um arquivo CSV.
        """
        rows = [
            ['generation'] + generations,
            ['best_reward'] + best_rewards,
            ['avg_reward'] + avg_rewards,
            ['sigma'] + sigmas
        ]
        with open(str(self.dir_logs / log_filename), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
