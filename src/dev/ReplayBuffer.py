import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.

    Methods
    -------
    add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
        Sample a batch of transitions from the buffer.
    __len__()
        Return the current size of the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = TensorDictReplayBuffer(
                storage=LazyMemmapStorage(
                capacity,
                device=self.device)
                )

    def add_sample(
        self,
        state,
        action,
        reward,
        new_state,
        done: bool,
    ):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state vector of the added transition.
        action : np.int64
            The action of the added transition.
        reward : float
            The reward of the added transition.
        next_state : np.ndarray
            The next state vector of the added transition.
        done : bool
            The final state of the added transition.
        """
        self.buffer.add(TensorDict({
                    "state": torch.tensor(state),
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "new_state": torch.tensor(new_state),
                    "done": torch.tensor(done)
                    }, batch_size=[]))

    def get_sample(
        self, batch_size: int
    ):
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, bool]
            A batch of `batch_size` transitions.
        """
        batch = self.buffer.sample(batch_size)
        states = batch.get('state').type(torch.FloatTensor).to(self.device)
        new_states = batch.get('new_state').type(torch.FloatTensor).to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        dones = batch.get('done').squeeze().to(self.device)
        return states, actions, rewards, new_states, dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            The current size of the buffer.
        """
        return self.buffer.__sizeof__()