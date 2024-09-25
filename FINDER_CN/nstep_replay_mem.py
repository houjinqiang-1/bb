from typing import List, Tuple
import random
from graph import Graph
from mvc_env import MvcEnv

#ReplaySample 类用于存储一批（batch）的经验样本。每个经验样本包括了状态序列、下一个状态序列、动作序列、奖励序列和终止状态标志序列。
class ReplaySample:
    def __init__(self, batch_size: int):
        # 存储批量的图对象
        self.g_list: List[Graph] = []
        # 存储状态序列
        self.list_st: List[List[int]] = []
        # 存储下一个状态序列
        self.list_s_primes: List[List[int]] = []
        # 存储动作序列
        self.list_at: List[int] = []
        # 存储奖励序列
        self.list_rt: List[float] = []
        # 存储终止状态标志序列
        self.list_term: List[bool] = []
        # 存储已删除的边：
        self.list_st_edges: List[List[set]] = []
        self.list_s_primes_edges: List[List[set]] = []
#NStepReplayMem 类实现了一个 N-步骤的经验回放内存，用于存储和管理经验样本。
class NStepReplayMem:
    def __init__(self, memory_size: int):
        # 内存大小
        self.memory_size = memory_size
        # 存储图对象的列表
        self.graphs: List[Graph] = [Graph()] * memory_size
        # 存储动作序列
        self.actions: List[int] = [0] * memory_size
        # 存储奖励序列
        self.rewards: List[float] = [0.0] * memory_size
        # 存储状态序列
        self.states: List[List[int]] = [[] for _ in range(memory_size)]
        # 存储下一个状态序列
        self.s_primes: List[List[int]] = [[] for _ in range(memory_size)]
        # 存储终止状态标志序列
        self.terminals: List[bool] = [False] * memory_size
        # 当前位置
        self.current = 0
        # 存储的样本数量
        self.count = 0
        #已删除的边
        self.remove_edges: List[List[set]] = [[set(), set()] for _ in range(memory_size)]
        self.remove_edges_primes: List[List[set]] = [[set(), set()] for _ in range(memory_size)]


    def add(self, g: Graph, s_t: List[int], a_t: int, r_t: float, s_prime: List[int], terminal: bool, remove_edges: List[set], remove_edges_primes: List[set]):
    #def add(self, g: Graph, s_t: List[int], a_t: int, r_t: float, s_prime: List[int], terminal: bool):
        # 向经验回放缓冲区中添加一个经验
        self.graphs[self.current] = g
        self.actions[self.current] = a_t
        self.rewards[self.current] = r_t
        self.states[self.current] = s_t.copy()
        self.s_primes[self.current] = s_prime.copy()
        self.terminals[self.current] = terminal
        self.remove_edges[self.current] = [remove_edges[0].copy(),remove_edges[1].copy()]
        self.remove_edges_primes[self.current] = [remove_edges_primes[0].copy(),remove_edges_primes[1].copy()]  
        # 更新存储的样本数量和位置
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def add_from_env(self, env: MvcEnv, n_step: int):
        #从环境中获取一系列的经验并添加到经验回放缓冲区中
        assert env.isTerminal()
        num_steps = len(env.state_seq)
        assert num_steps > 0
        env.sum_rewards[num_steps - 1] = env.reward_seq[num_steps - 1]
        for i in range(num_steps - 1, -1, -1):
            if i < num_steps - 1:
                env.sum_rewards[i] = env.sum_rewards[i + 1] + env.reward_seq[i]

        for i in range(num_steps):
            term_t = False
            cur_r = 0.0
            s_prime = []
            if i + n_step >= num_steps:
                cur_r = env.sum_rewards[i]
                s_prime = env.action_list.copy()
                remove_edges_primes = env.remove_edge.copy()
                term_t = True
            else:
                cur_r = env.sum_rewards[i] - env.sum_rewards[i + n_step]
                s_prime = env.state_seq[i + n_step].copy()
                remove_edges_primes = [env.state_seq_edges[i + n_step][0].copy(), env.state_seq_edges[i + n_step][1].copy()]
            self.add(env.graph, env.state_seq[i].copy(), env.act_seq[i], cur_r, s_prime, term_t, [env.state_seq_edges[i][0].copy(),env.state_seq_edges[i][1].copy()], remove_edges_primes)
            #self.add(env.graph, env.state_seq[i].copy(), env.act_seq[i], cur_r, s_prime, term_t)
            
    def sampling(self, batch_size: int) -> ReplaySample:
        assert self.count >= batch_size
        result = ReplaySample(batch_size)

        # 从存储的样本中随机抽取指定数量的样本
        indices = random.sample(range(self.count), batch_size)
        result.g_list = [self.graphs[i] for i in indices]
        result.list_st = [self.states[i].copy() for i in indices]
        result.list_at = [self.actions[i] for i in indices]
        result.list_rt = [self.rewards[i] for i in indices]
        result.list_s_primes = [self.s_primes[i].copy() for i in indices]
        result.list_term = [self.terminals[i] for i in indices]
        result.list_st_edges = [self.remove_edges[i].copy() for i in indices]
        result.list_s_primes_edges = [self.remove_edges_primes[i].copy() for i in indices]
        return result
