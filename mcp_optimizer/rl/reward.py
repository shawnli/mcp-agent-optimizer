"""
奖励函数设计 - 实现多种前沿奖励计算方法

包含:
1. 进程奖励模型(Process Reward Model, PRM)
2. 增量奖励(Progress-based Reward) - 用户提出的想法
3. 对比学习奖励(Contrastive Reward)
4. 优势函数(Advantage Function)
"""
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel
import numpy as np
from enum import Enum


class RewardType(str, Enum):
    """奖励类型"""
    OUTCOME = "outcome"  # 结果奖励(任务成功/失败)
    PROCESS = "process"  # 进程奖励(每一步的质量)
    PROGRESS = "progress"  # 进步奖励(相比上一步的改进)
    CONTRASTIVE = "contrastive"  # 对比奖励
    SHAPED = "shaped"  # 奖励塑形


class StateValue(BaseModel):
    """状态价值评估"""
    state_repr: str  # 状态的文本表示
    value: float  # 状态价值 V(s)
    confidence: float = 1.0  # 评估置信度


class ProgressReward:
    """
    进步奖励计算器
    
    核心思想:评估每一步action是否让状态向目标靠近
    reward = V(s_t) - V(s_t-1)
    
    这对应强化学习中的:
    - 优势函数(Advantage Function)
    - 时序差分(Temporal Difference)
    - 最新的AgentPRM研究方向
    """
    
    def __init__(
        self,
        value_estimator: Optional[Callable] = None,
        normalize: bool = True,
        clip_range: tuple = (-1.0, 1.0)
    ):
        """
        Args:
            value_estimator: 状态价值评估函数 V(s)
                如果为None,使用默认的启发式评估
            normalize: 是否归一化奖励
            clip_range: 奖励裁剪范围
        """
        self.value_estimator = value_estimator or self._default_value_estimator
        self.normalize = normalize
        self.clip_range = clip_range
        
        # 历史记录,用于归一化
        self.reward_history = []
    
    def compute_progress_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        goal: str
    ) -> float:
        """
        计算进步奖励
        
        Args:
            prev_state: 上一步的状态
            curr_state: 当前步的状态
            goal: 目标描述
        
        Returns:
            进步奖励值
        """
        # 评估两个状态的价值
        v_prev = self.value_estimator(prev_state, goal)
        v_curr = self.value_estimator(curr_state, goal)
        
        # 计算进步 = 当前价值 - 之前价值
        progress = v_curr - v_prev
        
        # 归一化
        if self.normalize:
            self.reward_history.append(progress)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            
            if len(self.reward_history) > 10:
                mean = np.mean(self.reward_history)
                std = np.std(self.reward_history) + 1e-8
                progress = (progress - mean) / std
        
        # 裁剪
        progress = np.clip(progress, self.clip_range[0], self.clip_range[1])
        
        return float(progress)
    
    def _default_value_estimator(
        self,
        state: Dict[str, Any],
        goal: str
    ) -> float:
        """
        默认的状态价值评估(启发式)
        
        实际应用中,这应该是:
        1. 训练好的神经网络价值函数
        2. LLM作为评判器
        3. 基于规则的距离度量
        """
        # 示例:基于状态中的"进度"字段
        if "progress" in state:
            return state["progress"]
        
        # 示例:基于已完成的子任务数量
        if "completed_subtasks" in state:
            total = state.get("total_subtasks", 1)
            completed = state["completed_subtasks"]
            return completed / total
        
        # 默认:无法评估,返回0
        return 0.0


class ProcessRewardModel:
    """
    进程奖励模型(PRM)
    
    基于AgentPRM论文(2025)的实现
    为每一步action提供细粒度的奖励信号
    """
    
    def __init__(
        self,
        model=None,  # 实际的PRM模型(如神经网络)
        use_monte_carlo: bool = True
    ):
        """
        Args:
            model: PRM模型,输入(state, action, context),输出Q值
            use_monte_carlo: 是否使用Monte Carlo rollouts来标注训练数据
        """
        self.model = model
        self.use_monte_carlo = use_monte_carlo
        
        # 存储训练数据
        self.training_data = []
    
    def evaluate_step(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        context: str
    ) -> float:
        """
        评估单步的质量
        
        Returns:
            Q(s, a): 该步骤的预期价值
        """
        if self.model is None:
            # 如果没有训练好的模型,使用启发式
            return self._heuristic_evaluation(state, action, context)
        
        # 使用训练好的模型
        return self.model.predict(state, action, context)
    
    def collect_rollout_data(
        self,
        trajectory: List[Dict[str, Any]],
        final_reward: float
    ):
        """
        收集Monte Carlo rollout数据
        
        Args:
            trajectory: 完整轨迹 [(state, action, reward), ...]
            final_reward: 最终结果奖励
        """
        # 反向传播奖励(Monte Carlo)
        returns = []
        G = final_reward
        gamma = 0.99  # 折扣因子
        
        for step in reversed(trajectory):
            returns.insert(0, G)
            G = step.get("reward", 0) + gamma * G
        
        # 存储为训练数据
        for i, step in enumerate(trajectory):
            self.training_data.append({
                "state": step["state"],
                "action": step["action"],
                "context": step.get("context", ""),
                "target_value": returns[i]
            })
    
    def train(self, epochs: int = 10):
        """训练PRM模型"""
        if self.model is None or len(self.training_data) == 0:
            return
        
        # 这里应该是实际的神经网络训练
        # 简化示例:
        print(f"Training PRM on {len(self.training_data)} samples for {epochs} epochs")
        # self.model.fit(self.training_data, epochs=epochs)
    
    def _heuristic_evaluation(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        context: str
    ) -> float:
        """启发式评估(占位符)"""
        return 0.0


class ContrastiveReward:
    """
    对比学习奖励
    
    基于2024年最新研究:
    通过对比"好的轨迹"和"坏的轨迹"来学习奖励
    """
    
    def __init__(
        self,
        baseline_policy=None,
        temperature: float = 1.0
    ):
        """
        Args:
            baseline_policy: 基线策略(用于生成对比样本)
            temperature: 对比学习的温度参数
        """
        self.baseline_policy = baseline_policy
        self.temperature = temperature
    
    def compute_contrastive_reward(
        self,
        current_trajectory: List[Dict],
        baseline_trajectory: List[Dict]
    ) -> List[float]:
        """
        计算对比奖励
        
        Args:
            current_trajectory: 当前策略的轨迹
            baseline_trajectory: 基线策略的轨迹
        
        Returns:
            每一步的对比奖励
        """
        rewards = []
        
        for i in range(min(len(current_trajectory), len(baseline_trajectory))):
            curr_step = current_trajectory[i]
            base_step = baseline_trajectory[i]
            
            # 简化:比较状态价值
            curr_value = curr_step.get("value", 0.0)
            base_value = base_step.get("value", 0.0)
            
            # 对比奖励 = 当前 - 基线
            contrastive_r = (curr_value - base_value) / self.temperature
            rewards.append(contrastive_r)
        
        return rewards


class RewardShaper:
    """
    奖励塑形(Reward Shaping)
    
    组合多种奖励信号:
    - 结果奖励(outcome)
    - 进步奖励(progress)
    - 效率惩罚(efficiency)
    - 安全约束(safety)
    """
    
    def __init__(
        self,
        outcome_weight: float = 1.0,
        progress_weight: float = 0.5,
        efficiency_weight: float = 0.1,
        safety_weight: float = 2.0
    ):
        self.outcome_weight = outcome_weight
        self.progress_weight = progress_weight
        self.efficiency_weight = efficiency_weight
        self.safety_weight = safety_weight
        
        self.progress_calculator = ProgressReward()
    
    def compute_shaped_reward(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        action: Dict[str, Any],
        goal: str,
        is_terminal: bool = False,
        outcome_reward: float = 0.0
    ) -> Dict[str, float]:
        """
        计算塑形后的综合奖励
        
        Returns:
            包含各组件和总奖励的字典
        """
        rewards = {}
        
        # 1. 结果奖励(只在终止时)
        if is_terminal:
            rewards["outcome"] = outcome_reward * self.outcome_weight
        else:
            rewards["outcome"] = 0.0
        
        # 2. 进步奖励(每一步)
        progress = self.progress_calculator.compute_progress_reward(
            prev_state, curr_state, goal
        )
        rewards["progress"] = progress * self.progress_weight
        
        # 3. 效率惩罚(鼓励用更少步骤)
        step_penalty = -0.01  # 每一步的小惩罚
        rewards["efficiency"] = step_penalty * self.efficiency_weight
        
        # 4. 安全约束(如果违反约束,大惩罚)
        safety_violation = curr_state.get("safety_violation", False)
        rewards["safety"] = -1.0 * self.safety_weight if safety_violation else 0.0
        
        # 总奖励
        rewards["total"] = sum(rewards.values())
        
        return rewards
