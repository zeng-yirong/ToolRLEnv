import gymnasium as gym
from gymnasium import spaces
from utils import FunctionCallExecutor, StateManager
from typing import Dict, Any, Tuple, Optional, Union
from Travel import TravelBookingEnv
from VehicleControl import VehicleControlEnv
from WebSearch import WebSearchEnv
from Twitter import TwitterEnv
from TradingBot import TradingBotEnv
from Ticket import TicketEnv
from Message import MessageEnv
from Math import MathEnv
from File import FileEnv
import numpy as np


class GeneralEnv(gym.Env):
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

        # 初始化执行器和状态管理器
        self.executor = FunctionCallExecutor(test_entry)
        self.state_manager = StateManager(test_entry)

        # 设置初始状态
        self._setup_initial_state()

        # 定义动作和观察空间
        self._setup_spaces()
        self._init_apis_()

    def _init_apis_(self):
        self.travel_env = TravelBookingEnv(test_entry=self.test_entry)
        self.vehicle_control_env = VehicleControlEnv(test_entry=self.test_entry)
        self.web_search_env = WebSearchEnv(test_entry=self.test_entry)
        self.twitter_env = TwitterEnv(test_entry = self.test_entry)
        self.trading_bot_env = TradingBotEnv(test_entry = self.test_entry)
        self.ticket_env = TicketEnv(test_entry = self.test_entry)
        self.message_env = MessageEnv(test_entry = self.test_entry)
        self.math_env = MathEnv(test_entry = self.test_entry)
        self.file_env = FileEnv(test_entry = self.test_entry)
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
        self.action_space = spaces.Discrete(len(self.function_docs) + 1)  # +1 for "no action"
        charset = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz .,!?;:-()")
        obs_space = {
            "question": spaces.Text(1000, charset=charset),  # 用户问题
            "function_docs": spaces.Sequence(  # 函数文档列表
                spaces.Dict({
                    "name": spaces.Text(100, charset=charset),
                    "description": spaces.Text(500, charset=charset),
                    "parameters": spaces.Dict({'type': spaces.Text(100,charset=charset),
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
                                               ),
                                               'env':spaces.Text(100, charset=charset)
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

    def _extract_function_name(self, function_call: str) -> str:
        """提取函数名"""
        if "(" in function_call:
            return function_call.split("(")[0].strip()
        return function_call.strip()

    def _extract_parameters(self, function_call: str) -> Dict[str, Any]:
        """从函数调用中提取参数"""
        params = {}
        if "(" in function_call and ")" in function_call:
            param_str = function_call.split("(")[1].split(")")[0]
            if "=" in param_str:
                for param in param_str.split(","):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')

                        # 尝试转换数据类型
                        if value.lower() in ["true", "false"]:
                            params[key] = value.lower() == "true"
                        elif value.isdigit():
                            params[key] = int(value)
                        elif self._is_float(value):
                            params[key] = float(value)
                        else:
                            params[key] = value
        return params

    def _is_float(self, value: str) -> bool:
        """检查字符串是否可以转换为浮点数"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _compute_reward(self, execution_success: bool, execution_result: str) -> float:
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
        if self.state_manager.is_goal_achieved() and self.current_turn == len(self.function_docs):
            self.task_completed = True
            self.done = True
        elif self.state_manager.is_failed():
            self.done = True

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        重置当前环境
        """
        super().reset(seed=seed)

        # 重置状态
        self.current_turn = 0
        self.execution_history = []
        self.reward_history = []
        self.done = False
        self.task_completed = False
        self._init_apis_()
        # 重新加载初始配置
        self.state_manager.reset()
        self.state_manager.load_initial_config(self.initial_config)
        return self._get_observation(), self._get_info()


    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
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
        function_call = self._parse_action(int(action))
        function_name = self._extract_function_name(function_call)
        function_param = self._extract_parameters(function_call)
        function_env = self.function_docs[action]['parameters']['env']
        if function_env == 'travel':
            execution_result, execution_success = self.travel_env.execute_function_call(function_name = function_name,parameters=function_param)
        elif function_env == 'vehicle_control':
            execution_result, execution_success = self.vehicle_control_env.execute_function_call(function_name=function_name,parameters=function_param)
        elif function_env == 'web_search':
            execution_result, execution_success = self.web_search_env.execute_function_call(function_name=function_name,parameters=function_param)
        elif function_env == 'twitter':
            execution_result, execution_success = self.twitter_env.execute_function_call(function_name=function_name,parameters=function_param)
        elif function_env == 'trading_bot':
            execution_result, execution_success = self.trading_bot_env.execute_function_call(function_name=function_name,parameters=function_param)
        elif function_env == 'ticket':
            execution_result, execution_success = self.ticket_env.execute_function_call(function_name=function_name,parameters=function_param)
        elif function_env == 'message':
            execution_result, execution_success = self.message_env.execute_function_call(function_name=function_name,parameters=function_param)
        elif function_env == 'math':
            execution_result, execution_success = self.math_env.execute_function_call(function_name=function_name,parameters=function_param)
        elif function_env == 'file':
            execution_result, execution_success = self.file_env.execute_function_call(function_name=function_name,parameters=function_param)
        else:
            execution_result = f"No env named {function_name}"
            execution_success = False
        print("env： "+str(function_env))
        print("execution_result: "+ str(execution_result))
        print("execution_success: " + str(execution_success))
        self._update_state(execution_result, execution_success)
        self._check_completion()
        reward = self._compute_reward(execution_success, execution_result)
        truncated = self.current_turn >= self.max_turns
        return self._get_observation(), reward, self.done, truncated, self._get_info()


    def _parse_action(self, action:int) -> str:
        """解析动作为函数调用字符串"""

        if isinstance(action, int):
            if action == len(self.function_docs):
                return ""
            # 简化处理：返回函数名，实际中需要参数生成
            func_doc = self.function_docs[action]
            para = ""
            for item in func_doc['parameters']['properties']:
                para += f",{item['properties_key']}={item['properties_value']['value']}"
            return f"{func_doc['name']}({para[1:]})"
        else:

            raise ValueError(f"Unsupported action type: {type(action)}")
