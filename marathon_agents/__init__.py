"""
marathon_agents — one file per algorithm, all exported from here.
Import in marathon.py with:
    from marathon_agents import RandomAgent, BFSAgent, AStarAgent, HamiltonianAgent, DQNAgent
"""
from .random_agent      import RandomAgent
from .bfs_agent         import BFSAgent
from .astar_agent       import AStarAgent
from .hamiltonian_agent import HamiltonianAgent
from .dqn_agent         import DQNAgent

__all__ = ['RandomAgent', 'BFSAgent', 'AStarAgent', 'HamiltonianAgent', 'DQNAgent']
