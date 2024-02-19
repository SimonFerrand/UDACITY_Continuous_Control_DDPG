import numpy as np
import random
from numpy.random import default_rng
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

import config
from model import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed, episodes):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int) : agent number
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)      # random initialization with the seed
        self.episodes = episodes
        
        # Number of parallel agents
        self.n_agents = n_agents        

        # Hyperparameters
            # Default
        self.BUFFER_SIZE = config.BUFFER_SIZE      # replay buffer size 
        self.BATCH_SIZE = config.BATCH_SIZE        # minibatch size 
        self.GAMMA = config.GAMMA                  # discount factor 
        self.TAU = config.TAU                      # for soft update of target parameters
        self.LR_ACTOR = config.LR_ACTOR            # learning rate of the actor
        self.LR_CRITIC = config.LR_CRITIC          # learning rate of the critic 
        self.WEIGHT_DECAY = config.WEIGHT_DECAY    # L2 weight decay
            # Noise Exploration (exponential reduction) 
        self.NOISE_SIGMA_INITIAL = config.NOISE_SIGMA_INITIAL # Initial sigma value (higher= greater exploration)
        self.NOISE_DECAY_RATE = config.NOISE_DECAY_RATE       # Exponential decrease noise over time 
        self.NOISE_SIGMA_MIN = config.NOISE_SIGMA_MIN         # Minimum sigma value to prevent noise from disappearing completely
            # Sample and Learn successively
        self.LEARNING_ITERATIONS = config.LEARNING_ITERATIONS # Number of times the agent learn at each time step, based on samples  taken from the replay memory
            # For Prioritized Experience Replay (PER) only
        self.PER_ALPHA = config.PER_ALPHA                                              # controls the degree of prioritization used in Prioritized Experience Replay (PER)
        self.PER_BETA_INITIAL = config.PER_BETA_INITIAL                                # Initial beta value
        self.PER_BETA_MAX =config.PER_BETA_MAX                                         # Maximum beta value # 1.0
        self.PER_BETA_N_EPISODES_MAX = config.PER_BETA_N_EPISODES_MAX                  # Number of episodes to reach BETA max
        self.beta = self.PER_BETA_INITIAL                                              # BETA value on initialization
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)

        # Set same weights for both local and target networks (because Tau =1) performed once when creating an instance of the Agent class = hardcopy 
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.soft_update(self.critic_local, self.critic_target, 1)

        # Noise process (Exploration)
        # self.noise = OUNoise(action_size, random_seed, noise_decay=self.NOISE_DECAY, noise_sigma=self.NOISE_SIGMA) 
        self.noise = OUNoise(action_size, random_seed, noise_sigma_initial=self.NOISE_SIGMA_INITIAL, noise_decay_rate=self.NOISE_DECAY_RATE, min_noise_sigma=self.NOISE_SIGMA_MIN) 
        
        # Initialize Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed, self.PER_ALPHA)

        # Monitoring indicators
        self.last_critic_loss = None
        self.last_actor_loss = None
        
        self.beta_history = []
        self.weight_magnitudes = []
        self.td_errors_history = []

        
    def step_memory_and_learn(self, states, actions, rewards, next_states, dones):
        """ Given a batch of S,A,R,S' experiences, it saves them into the
            experience buffer, and occasionally samples from the experience
            buffer to perform training steps.
        """
        
        # Save experiences with an initial high priority to ensure they are sampled at least once.
        for i in range(self.n_agents):
            # Get the current max priority in the buffer
            max_priority = self.memory.max_priority if hasattr(self.memory, 'max_priority') else 1.0
            # Add experience in the memory
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i], max_priority)
    
        # Learning step if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            for _ in range(self.LEARNING_ITERATIONS):
                 # Sample from the memory to get states, actions, rewards, next_states, dones, indices, and weights
                states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.beta) 
                experiences = (states, actions, rewards, next_states, dones)
                self.learn(experiences, self.GAMMA, indices, weights)
                # Beta update at each learning
                #self.beta = min(self.PER_BETA_MAX, self.beta + self.PER_BETA_INCREMENT_PER_SAMPLING)
                            
    def act(self, states, add_noise=True):
        """Returns actions for given states as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += [self.noise.sample() for _ in range(self.n_agents)]
        return np.clip(actions, -1, 1)                 # action range form -1 to 1

    def reset(self, i_episode):
        # Noise update
        self.noise.reset(i_episode)
        # Beta PER update
        beta_increment_per_episode = (self.PER_BETA_MAX - self.PER_BETA_INITIAL) / self.PER_BETA_N_EPISODES_MAX
        self.beta = min(self.PER_BETA_MAX, self.PER_BETA_INITIAL + i_episode * beta_increment_per_episode)
        self.beta_history.append(self.beta)

    def learn(self, experiences, gamma, indices, weights):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
    
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            indices (list): list of indices of sampled experiences
            weights (torch.Tensor): tensor of importance sampling weights for each experience
        """
        states, actions, rewards, next_states, dones = experiences
    
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss using importance sampling weights
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets) * weights  # Adjust loss with weights
        critic_loss = critic_loss.mean()  # Take mean to reduce to a single scalar value                 
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.last_critic_loss = critic_loss.item()  # get critic loss
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.last_actor_loss = actor_loss.item()  # get actor loss
    
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)
    
        # ----------------------- update priorities in buffer ------------------ #
        # Calculate TD-Errors for updating priorities
        td_errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-5)
        
        # record weights
        self.weight_magnitudes.append(weights.mean().item())
        # record td_errors
        self.td_errors_history.append(np.mean(td_errors))

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def record_weights_and_gradients(self, i_episode):
        """Record weights and gradients data after each episode."""
        self.weights_data_episode = {'episode': i_episode, 'actor_weights': [], 'actor_gradients': [], 'critic_weights': [], 'critic_gradients': []}
        
        # Save data for actor
        for name, param in self.actor_local.named_parameters():
            if param.requires_grad:
                self.weights_data_episode['actor_weights'].append({
                    'layer': name,
                    'weight_mean': param.data.mean().item(),
                    'weight_std': param.data.std().item(),
                })
                if param.grad is not None:
                    self.weights_data_episode['actor_gradients'].append({
                        'layer': name,
                        'grad_mean': param.grad.data.mean().item(),
                        'grad_std': param.grad.data.std().item(),
                    })

        # Save data for critic
        for name, param in self.critic_local.named_parameters():
            if param.requires_grad:
                self.weights_data_episode['critic_weights'].append({
                    'layer': name,
                    'weight_mean': param.data.mean().item(),
                    'weight_std': param.data.std().item(),
                })
                if param.grad is not None:
                    self.weights_data_episode['critic_gradients'].append({
                        'layer': name,
                        'grad_mean': param.grad.data.mean().item(),
                        'grad_std': param.grad.data.std().item(),
                    })

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, noise_sigma_initial, noise_decay_rate, min_noise_sigma, mu=0., theta=0.15):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)          # The long-term mean towards which the noise process tends. Typically, this parameter is set to 0 for centered noise.    
        self.theta = theta                    # The rate of return to the mean. A higher θ results in more rapidly correlated noise.
        #self.sigma = noise_sigma             # Process volatility. A higher σ increases the amplitude of noise variations, enabling greater exploration.
        self.sigma = 0
        #self.noise_decay = noise_decay       # new parameter added that reduces sigma with each reset call to reduce exploration and promote exploitation over time (test SF)
        self.seed = random.seed(seed)
        self.noise_sigma_initial = noise_sigma_initial
        self.noise_decay_rate = noise_decay_rate
        self.min_noise_sigma = min_noise_sigma

    def reset(self,i_episode):
        """Reset the internal state (= noise) to mean (mu) and apply decay."""
        self.state = copy.copy(self.mu)
        #self.sigma *= self.noise_decay       # At the end of each episode, when agent.noise.reset() is called, the noise decreases
        self.sigma = max(self.noise_sigma_initial * (self.noise_decay_rate ** i_episode), self.min_noise_sigma) # This value would be updated at the end of each episode

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class PrioritizedReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """
        Initializes a Prioritized Replay Buffer.
        The Prioritized Experience Replay (PER) mechanism allows the agent to learn more efficiently by sampling experiences based on their importance, which is determined by their prediction error (TD-Error). This approach prioritizes experiences from which the agent can learn the most, thereby accelerating the learning process.
        
        Params
        ======
            buffer_size (int): maximum size of buffer. This defines the total number of experiences that can be stored in the buffer.
            batch_size (int): size of each training batch. This determines how many experiences are sampled from the buffer for each learning step.
            alpha (float): controls the degree of prioritization used. A value of 0 corresponds to uniform random sampling (no prioritization), while a value closer to 1 increases the focus on high-error experiences. Alpha helps to balance the exploration of the experience space (alpha near 0) with focused learning on high-priority experiences (alpha near 1).
        
        Experiences with higher TD-Errors are deemed more important as they indicate significant discrepancies between the agent's predictions and the actual outcomes. By focusing on these experiences, the agent can correct its predictions more effectively, leading to faster improvement. To counterbalance the sampling bias introduced by prioritization, Importance Sampling Weights are used during the learning updates, ensuring that the learning process remains unbiased while benefiting from the efficiency of prioritized replay.

       This implementation of PrioritizedReplayBuffer integrates priority management based on TD errors (TD-Error) of experiments and their use for weighted sampling of experiments during learning. Importance sampling weights are calculated and adjusted according to the beta parameter, which can be increased over time to reduce the bias introduced by prioritized sampling.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.priority_memory = deque(maxlen=buffer_size)  # Store priorities
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.alpha = alpha                                # Degree of prioritization
        self.buffer_size = buffer_size
        self.epsilon = 1e-5                               # Small amount to avoid zero priority
        self.max_priority = 1.0                           # Set max_priority to 1.0 or another high default value
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed                                  # numpy is better then random (self.seed = random.seed(seed)) 
        self.rng = default_rng(seed)                      # The performance difference between random and np.random.choice can be significant for large datasets or frequent operations, mainly
        
    def add(self, state, action, reward, next_state, done, td_error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        priority = (np.abs(td_error) + self.epsilon) ** self.alpha
        self.priority_memory.append(float(priority))
        self.max_priority = max(self.max_priority, priority)  
        """max_priority is dynamic and adapts according to the prediction errors (TD-errors) of experiments added to the buffer. It is used to ensure that newly added experiments with significant errors receive a high priority for sampling, thus favoring their selection for learning and potentially speeding up the learning process by focusing on the most "informative" experiments."""
        
    def sample(self, beta):
        """Sample a batch of experiences from memory based on their priorities."""
        priorities = np.array(list(self.priority_memory), dtype=np.float64).flatten()
        
        # Calculate selection probabilities based on priorities
        total_priority = np.sum(priorities)
        probabilities = priorities / total_priority
        
        # Select the indices of the experiments to be sampled according to probabilities
        indices = self.rng.choice(len(self.memory), size=self.batch_size, p=probabilities)
        
        # # Collect sampled experiments
        experiences = [self.memory[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Standardize weights to stabilize learning
        
        # Convert experiment data into PyTorch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions based on new TD-Errors."""
        for idx, priority in zip(indices, priorities):
            assert 0 <= idx < len(self.memory), "Index out of bounds."
            # Calculate the new priority and make sure it is increased by epsilon to avoid zero priorities.
            new_priority = (np.abs(priority) + self.epsilon) ** self.alpha
            # Make sure the index is valid and update the priority
            self.priority_memory[idx] = new_priority
            # Update max_priority if the new priority is higher
            self.max_priority = max(self.max_priority, new_priority)
           
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


'''class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)'''