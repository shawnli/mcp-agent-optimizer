"""
Group Relative Policy Optimization (GRPO)

基于VisTA论文和AgentPRM的强化学习算法实现
结合了:
1. PPO (Proximal Policy Optimization)
2. Group-based normalization
3. Process Reward Model
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RolloutBatch:
    """Rollout批次数据"""
    states: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    advantages: List[float]
    returns: List[float]


class GRPO:
    """
    Group Relative Policy Optimization
    
    核心改进:
    1. 使用group-based advantage normalization
    2. 支持process reward
    3. 裁剪重要性比率防止训练不稳定
    """
    
    def __init__(
        self,
        policy_model,
        value_model,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        group_size: int = 8
    ):
        """
        Args:
            policy_model: 策略网络 π(a|s)
            value_model: 价值网络 V(s)
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_ratio: PPO裁剪比率
            value_clip: 价值函数裁剪
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪
            group_size: 组大小(用于group normalization)
        """
        self.policy_model = policy_model
        self.value_model = value_model
        
        self.lr = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.group_size = group_size
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_value: float = 0.0
    ) -> Tuple[List[float], List[float]]:
        """
        计算Generalized Advantage Estimation (GAE)
        
        GAE结合了:
        - TD(0): 低方差,高偏差
        - Monte Carlo: 高方差,低偏差
        
        Args:
            rewards: 每一步的奖励
            values: 每一步的状态价值V(s)
            next_value: 最后一个状态的后继价值
        
        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_val = next_value
        
        # 反向计算
        for t in reversed(range(len(rewards))):
            # TD误差: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            # 这正是"当前步相比上一步的进步"!
            delta = rewards[t] + self.gamma * next_val - values[t]
            
            # GAE: A_t = δ_t + (γλ)*δ_{t+1} + (γλ)^2*δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            next_val = values[t]
        
        return advantages, returns
    
    def normalize_advantages(
        self,
        advantages: List[float],
        group_based: bool = True
    ) -> List[float]:
        """
        优势函数归一化
        
        Args:
            advantages: 原始优势值
            group_based: 是否使用group-based归一化(GRPO的核心)
        
        Returns:
            归一化后的优势值
        """
        advantages = np.array(advantages)
        
        if group_based and len(advantages) >= self.group_size:
            # Group-based normalization
            # 将数据分组,每组内独立归一化
            normalized = []
            
            for i in range(0, len(advantages), self.group_size):
                group = advantages[i:i + self.group_size]
                
                if len(group) > 1:
                    mean = group.mean()
                    std = group.std() + 1e-8
                    group_norm = (group - mean) / std
                else:
                    group_norm = group
                
                normalized.extend(group_norm)
            
            return normalized
        else:
            # 全局归一化
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            return ((advantages - mean) / std).tolist()
    
    def compute_policy_loss(
        self,
        old_log_probs: List[float],
        new_log_probs: List[float],
        advantages: List[float]
    ) -> float:
        """
        计算PPO策略损失(带裁剪)
        
        L^{CLIP}(θ) = E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)]
        
        其中 r(θ) = π_θ(a|s) / π_θ_old(a|s) 是重要性比率
        """
        old_log_probs = np.array(old_log_probs)
        new_log_probs = np.array(new_log_probs)
        advantages = np.array(advantages)
        
        # 重要性比率
        ratio = np.exp(new_log_probs - old_log_probs)
        
        # 裁剪后的比率
        ratio_clipped = np.clip(
            ratio,
            1.0 - self.clip_ratio,
            1.0 + self.clip_ratio
        )
        
        # PPO损失
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        policy_loss = -np.minimum(surr1, surr2).mean()
        
        return policy_loss
    
    def compute_value_loss(
        self,
        old_values: List[float],
        new_values: List[float],
        returns: List[float]
    ) -> float:
        """
        计算价值函数损失(带裁剪)
        
        防止价值函数更新过大
        """
        old_values = np.array(old_values)
        new_values = np.array(new_values)
        returns = np.array(returns)
        
        # 裁剪后的价值
        values_clipped = old_values + np.clip(
            new_values - old_values,
            -self.value_clip,
            self.value_clip
        )
        
        # 价值损失
        loss1 = (new_values - returns) ** 2
        loss2 = (values_clipped - returns) ** 2
        value_loss = np.maximum(loss1, loss2).mean()
        
        return value_loss
    
    def update(
        self,
        batch: RolloutBatch,
        num_epochs: int = 4
    ) -> Dict[str, float]:
        """
        执行一次策略更新
        
        Args:
            batch: Rollout数据批次
            num_epochs: 在同一批数据上训练的轮数
        
        Returns:
            训练指标
        """
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0
        }
        
        # 归一化优势
        normalized_advantages = self.normalize_advantages(
            batch.advantages,
            group_based=True
        )
        
        # 多轮训练
        for epoch in range(num_epochs):
            # 重新计算log_probs和values
            # (实际实现中需要调用模型)
            new_log_probs = batch.log_probs  # 占位符
            new_values = batch.values  # 占位符
            
            # 计算损失
            policy_loss = self.compute_policy_loss(
                batch.log_probs,
                new_log_probs,
                normalized_advantages
            )
            
            value_loss = self.compute_value_loss(
                batch.values,
                new_values,
                batch.returns
            )
            
            # 熵正则化(鼓励探索)
            entropy = 0.01  # 占位符,实际需要从策略分布计算
            
            # 总损失
            total_loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )
            
            # 更新指标
            metrics["policy_loss"] += policy_loss
            metrics["value_loss"] += value_loss
            metrics["entropy"] += entropy
            metrics["total_loss"] += total_loss
            
            # 梯度下降(占位符)
            # optimizer.zero_grad()
            # total_loss.backward()
            # clip_grad_norm_(parameters, self.max_grad_norm)
            # optimizer.step()
        
        # 平均指标
        for key in metrics:
            metrics[key] /= num_epochs
        
        return metrics
    
    def train_iteration(
        self,
        rollouts: List[List[Dict[str, Any]]],
        prm_model=None
    ) -> Dict[str, Any]:
        """
        一次完整的训练迭代
        
        对应AgentPRM的三阶段:
        1. Rollout: 收集轨迹
        2. Train PRM: 训练进程奖励模型
        3. Update Policy: 更新策略
        
        Args:
            rollouts: 多条完整轨迹
            prm_model: 进程奖励模型(可选)
        
        Returns:
            训练统计信息
        """
        all_batches = []
        
        # 处理每条轨迹
        for trajectory in rollouts:
            states = [step["state"] for step in trajectory]
            actions = [step["action"] for step in trajectory]
            rewards = [step["reward"] for step in trajectory]
            
            # 计算价值(使用PRM或价值网络)
            if prm_model:
                values = [
                    prm_model.evaluate_step(s, a, "")
                    for s, a in zip(states, actions)
                ]
            else:
                values = [0.0] * len(states)  # 占位符
            
            # 计算GAE
            advantages, returns = self.compute_gae(rewards, values)
            
            # 创建批次
            batch = RolloutBatch(
                states=states,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=[0.0] * len(states),  # 占位符
                advantages=advantages,
                returns=returns
            )
            
            all_batches.append(batch)
        
        # 更新策略
        total_metrics = {}
        for batch in all_batches:
            metrics = self.update(batch)
            
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
        
        # 平均指标
        for key in total_metrics:
            total_metrics[key] /= len(all_batches)
        
        return total_metrics
