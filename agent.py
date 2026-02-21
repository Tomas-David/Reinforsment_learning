import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# 1. Samotná neuronová síť
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# 2. RL Agent, který řídí učení
class DQNAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0 # Míra průzkumu (1.0 = 100% náhodné akce na začátku)
        self.epsilon_decay = 0.995 # Jak rychle bude snižovat náhodnost
        self.epsilon_min = 0.01 # Minimální náhodnost (aby občas zkusil něco nového)
        self.gamma = 0.9 # Slevový faktor (jak moc mu záleží na budoucích odměnách)
        
        self.memory = deque(maxlen=100000) # Paměť na poslední kroky
        
        # Vstup: 5 senzorů, Skrytá vrstva: 64 neuronů, Výstup: 3 akce
        self.model = Linear_QNet(5, 64, 3) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

    def get_state_tensor(self, state):
        return torch.tensor(state, dtype=torch.float)

    def remember(self, state, action, reward, next_state, done):
        # Uložení zkušenosti do paměti
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        # Na začátku dělá náhodné kroky, postupně víc a víc využívá naučenou síť
        if random.random() < self.epsilon:
            move = random.randint(0, 2) # 0: rovně, 1: doleva, 2: doprava
        else:
            state_tensor = self.get_state_tensor(state)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item() # Vybere akci s nejvyšším skóre
        return move

    def train_experience_replay(self, batch_size):
        # Pokud nemá dost zkušeností, ještě se neučí
        if len(self.memory) < batch_size:
            return

        # Vybere náhodný vzorek z paměti
        mini_sample = random.sample(self.memory, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # Q-Learning vzorec
        # 1. Zjistíme, co si síť myslí teď
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 2. Zjistíme, jaká je maximální hodnota v dalším kroku
        max_next_q = self.model(next_states).max(1)[0]
        
        # 3. Vypočítáme, co by měla být správná hodnota (Bellmanova rovnice)
        expected_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 4. Opravíme síť (zpětná propagace)
        loss = self.criterion(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Snižování epsilon (aby méně prozkoumával a více hrál)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay