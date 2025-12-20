"""
BFCL函数调用环境的Gymnasium基类

将BFCL的多轮函数调用评估框架适配到Gymnasium环境，
提供标准的强化学习接口。
"""

import copy
import json
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

# 处理相对导入问题


from utils import FunctionCallExecutor, StateManager



class FunctionCallingEnv(gym.Env):
    """
    BFCL函数调用环境的Gymnasium基类

    该环境将BFCL的多轮函数调用任务转换为强化学习问题，
    代理通过选择函数调用来完成用户请求。
    """

    def __init__(
        self,
        test_entry: Dict[str, Any],
        max_turns: int = 10,
        task_type: str = "task_type",
        reward_config: Optional[Dict[str, float]] = None
    ):
        """
        初始化函数调用环境

        Args:
            test_entry: BFCL测试条目，包含问题、函数文档等
            max_turns: 最大交互轮数
            reward_config: 奖励配置参数
        """
        super().__init__()

        self.test_entry = test_entry
        self.max_turns = max_turns
        self.current_turn = 0
        self.task_type = task_type
        # 默认奖励配置
        self.reward_config = {
            "correct_function_call": 1.0,
            "correct_state": 2.0,
            "incorrect_function_call": -0.5,
            "state_mismatch": -1.0,
            "turn_penalty": -0.1,
            "task_completion": 5.0,
            "task_failure": -2.0
        }
        if reward_config:
            self.reward_config.update(reward_config)

        # 初始化执行器和状态管理器
        self.executor = FunctionCallExecutor(test_entry)
        self.state_manager = StateManager(test_entry)

        # 设置初始状态
        self._setup_initial_state()

        # 定义动作和观察空间
        self._setup_spaces()

    def _setup_initial_state(self):
        """设置初始状态"""
        self.current_question = self.test_entry["question"][0][0]["content"]
        self.function_docs = self.test_entry["function"]
        self.involved_classes = self.test_entry.get("involved_classes", [])
        self.initial_config = self.test_entry.get("initial_config", {})

        # 加载初始配置到状态管理器
        self.state_manager.load_initial_config(self.initial_config)

        # 执行历史记录
        self.execution_history = []
        self.reward_history = []
        self.done = False
        self.task_completed = False

    def _setup_spaces(self):
        """设置动作和观察空间"""
        # 动作空间：选择要调用的函数
        # 这里简化为离散动作，实际中可能需要更复杂的动作表示
        self.action_space = spaces.Discrete(len(self.function_docs) + 1)  # +1 for "no action"
        charset = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz .,!?;:-()")
        # 观察空间：包含问题、函数文档、当前状态等
        obs_space = {
            "question": spaces.Text(1000, charset=charset),  # 用户问题
            "function_docs": spaces.Sequence(  # 函数文档列表
                spaces.Dict({
                    "name": spaces.Text(100, charset=charset),
                    "description": spaces.Text(500, charset=charset),
                    "parameters": spaces.Dict({'type': spaces.Text(100),
                                               'properties':spaces.Sequence(
                                                   spaces.Dict({
                                                       'properties_key': spaces.Text(100,charset=charset),
                                                       'properties_value':
                                                           spaces.Dict({
                                                               'type':spaces.Text(min_length=0, max_length=100,charset=charset),
                                                               'description': spaces.Text(min_length=0, max_length=100,charset=charset),
                                                               'value': spaces.Text(min_length=0, max_length=100, charset=charset),
                                                           })
                                                   })
                                               ),
                                               'required':spaces.Sequence(
                                                   spaces.Text(100, charset=charset),
                                               )
                    })
                }),
            ),
            "current_state": spaces.Dict({  # 当前环境状态
                "turn": spaces.Discrete(self.max_turns + 1),
                "execution_history": spaces.Sequence(spaces.Text(min_length=0, max_length=500, charset=charset)),
            }),
            "last_execution_result": spaces.Text(
                min_length=0,
                max_length=1000,
                charset=charset,
            ),  # 上次执行结果
            "available_actions": spaces.Sequence(  # 可用动作列表
                spaces.Text(100, charset=charset),
            )
        }

        self.observation_space = spaces.Dict(obs_space)


    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """重置环境到初始状态"""
        super().reset(seed=seed)

        # 重置状态
        self.current_turn = 0
        self.execution_history = []
        self.reward_history = []
        self.done = False
        self.task_completed = False

        # 重新加载初始配置
        self.state_manager.reset()
        self.state_manager.load_initial_config(self.initial_config)

        return self._get_observation(), self._get_info()

    def step(self, action: Union[int, str, Dict]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 代理选择的动作，可以是函数调用索引、函数调用字符串或字典

        Returns:
            observation: 新的观察状态
            reward: 奖励信号
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before continuing.")

        self.current_turn += 1

        # 解析动作
        function_call = self._parse_action(action)

        # 执行函数调用
        execution_result, execution_success = self._execute_function_call(function_call)

        # 计算奖励
        reward = self._compute_reward(execution_success, execution_result)

        # 更新状态
        self._update_state(execution_result, execution_success)

        # 检查是否完成
        self._check_completion()

        # 检查是否超出最大轮数
        truncated = self.current_turn >= self.max_turns

        return self._get_observation(), reward, self.done, truncated, self._get_info()

    def _parse_action(self, action: Union[int, str, Dict]) -> str:
        """解析动作为函数调用字符串"""
        # 处理numpy整数类型
        if hasattr(action, 'dtype') and hasattr(action, 'item'):
            action = action.item()

        if isinstance(action, int):
            if action == len(self.function_docs):  # No action
                return ""
            # 简化处理：返回函数名，实际中需要参数生成
            func_doc = self.function_docs[action]
            para = ""
            for item in func_doc['parameters']['properties']:
                para += f",{item['properties_key']}={item['properties_value']['value']}"
            return f"{func_doc['name']}({para[1:]})"
        elif isinstance(action, str):
            return action
        elif isinstance(action, dict):
            # 将字典转换为函数调用字符串
            return self._dict_to_function_call(action)
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def _dict_to_function_call(self, action_dict: Dict) -> str:
        """将动作字典转换为函数调用字符串"""
        func_name = action_dict.get("function", "")
        params = action_dict.get("parameters", {})

        if not func_name:
            return ""

        param_strs = []
        for key, value in params.items():
            if isinstance(value, str):
                param_strs.append(f'{key}="{value}"')
            else:
                param_strs.append(f'{key}={value}')

        return f"{func_name}({', '.join(param_strs)})"

    def _execute_function_call(self, function_call: str) -> Tuple[str, bool]:
        """执行函数调用"""
        if not function_call:
            return "No function executed", True

        try:
            # 使用执行器执行函数调用
            result = self.executor.execute([function_call])
            execution_result = result[0] if result else "No result"
            success = True
        except Exception as e:
            execution_result = f"Error: {str(e)}"
            success = False

        # 记录执行历史
        self.execution_history.append({
            "turn": self.current_turn,
            "function_call": function_call,
            "result": execution_result,
            "success": success
        })

        return execution_result, success

    def _compute_reward(self, execution_success: bool, execution_result: str) -> float:
        """计算奖励信号"""
        reward = 0.0

        # 基础奖励/惩罚
        if execution_success:
            reward += self.reward_config["correct_function_call"]
        else:
            reward += self.reward_config["incorrect_function_call"]

        # 状态检查奖励
        state_match = self.state_manager.check_state_consistency()
        if state_match:
            reward += self.reward_config["correct_state"]
        else:
            reward += self.reward_config["state_mismatch"]

        # 轮次惩罚
        reward += self.reward_config["turn_penalty"]

        # 任务完成奖励/惩罚
        if self.task_completed:
            reward += self.reward_config["task_completion"]
        elif self.done and not self.task_completed:
            reward += self.reward_config["task_failure"]

        self.reward_history.append(reward)
        return reward

    def _update_state(self, execution_result: str, execution_success: bool):
        """更新环境状态"""
        # 更新状态管理器
        self.state_manager.update_from_execution(execution_result, execution_success)

    def _check_completion(self):
        """检查任务是否完成"""
        # 检查是否达到目标状态
        if self.state_manager.is_goal_achieved():
            self.task_completed = True
            # self.done = True
        elif self.state_manager.is_failed():
            self.done = True

    def _get_observation(self) -> Dict:
        """获取当前观察状态"""
        return {
            "available_actions": tuple([func["name"] for func in self.function_docs]),
            "current_state": {
                "turn": self.current_turn,
                "execution_history": tuple([str(item) for item in self.execution_history[-5:]])  # 最近5次
            },
            "function_docs": tuple(self.function_docs),
            "last_execution_result": self.execution_history[-1]["result"] if self.execution_history else "",
            "question": self.current_question
        }

    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "task_completed": self.task_completed,
            "total_reward": sum(self.reward_history),
            "execution_history": self.execution_history,
            "state_consistency": self.state_manager.check_state_consistency()
        }

    def render(self):
        """渲染环境状态"""
        print(f"=== Turn {self.current_turn}/{self.max_turns} ===")
        print(f"Question: {self.current_question}")
        print(f"Task Completed: {self.task_completed}")
        if self.execution_history:
            last_exec = self.execution_history[-1]
            print(f"Last Action: {last_exec['function_call']}")
            print(f"Last Result: {last_exec['result']}")
        print(f"Total Reward: {sum(self.reward_history):.2f}")
        print("-" * 50)