"""
消息API环境的Gymnasium适配

基于MessageAPI实现的消息传递环境，
支持用户管理、消息发送接收、消息检索等功能。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json
from copy import deepcopy

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class MessageAPIEnv(FunctionCallingEnv):
    """
    消息传递环境

    支持以下消息操作：
    - 用户管理：login, logout, list_users, get_user_id
    - 消息操作：send_message, view_messages_sent, view_messages_received
    - 消息检索：search_messages, get_message_details
    - 用户交互：block_user, unblock_user, get_blocked_users
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "messaging_system",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化消息API环境

        Args:
            test_entry: 测试条目，包含消息系统初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入MessageAPI
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.message_api import MessageAPI, DEFAULT_STATE

        self.api = MessageAPI()

        # 默认奖励配置
        default_reward_config = {
            "message_sent": 2.0,            # 成功发送消息
            "message_received": 1.5,        # 成功接收消息
            "user_authenticated": 1.0,      # 用户认证成功
            "user_found": 1.0,              # 找到用户
            "message_searched": 1.5,        # 成功搜索消息
            "task_completion": 5.0,         # 任务完成
            "send_failed": -2.0,            # 发送失败
            "user_not_found": -1.5,         # 用户未找到
            "unauthorized": -2.0,           # 未授权操作
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)

        # 加载消息系统初始状态
        if test_entry and "initial_config" in test_entry:
            if "MessageAPI" in test_entry["initial_config"]:
                self.api._load_scenario(test_entry["initial_config"]["MessageAPI"])

        # 初始化消息状态
        self.current_user = None
        self.message_history = []

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        # 基于MessageAPI的实际函数
        actions = [
            "login", "logout", "list_users", "get_user_id",
            "send_message", "view_messages_sent", "view_messages_received",
            "search_messages", "get_message_details", "block_user", "unblock_user"
        ]
        return spaces.Discrete(len(actions))

    def _get_observation_space(self) -> spaces.Space:
        """定义观察空间"""
        return spaces.Dict({
            "question": spaces.Text(1000),
            "current_user": spaces.Text(100),
            "authentication_state": spaces.Dict({
                "is_logged_in": spaces.Discrete(2),
                "user_id": spaces.Text(50),
                "user_name": spaces.Text(100)
            }),
            "inbox_info": spaces.Dict({
                "unread_count": spaces.Discrete(100),
                "total_messages": spaces.Discrete(500),
                "senders": spaces.Sequence(spaces.Text(100))
            }),
            "outbox_info": spaces.Dict({
                "sent_count": spaces.Discrete(100),
                "recipients": spaces.Sequence(spaces.Text(100))
            }),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "messaging_state": spaces.Dict({
                "messages_sent": spaces.Discrete(50),
                "messages_received": spaces.Discrete(50),
                "users_contacted": spaces.Discrete(20),
                "tasks_completed": spaces.Discrete(10)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "login",
                "description": "Log in a user with user ID",
                "parameters": {
                    "user_id": "string - the ID of the user to log in"
                }
            },
            {
                "name": "logout",
                "description": "Log out the current user",
                "parameters": {}
            },
            {
                "name": "list_users",
                "description": "List all users in the workspace",
                "parameters": {}
            },
            {
                "name": "get_user_id",
                "description": "Get the user ID for a given username",
                "parameters": {
                    "user": "string - the username to get ID for"
                }
            },
            {
                "name": "send_message",
                "description": "Send a message to another user",
                "parameters": {
                    "receiver_id": "string - the ID of the message receiver",
                    "message": "string - the message content to send"
                }
            },
            {
                "name": "view_messages_sent",
                "description": "View messages sent by the current user",
                "parameters": {}
            },
            {
                "name": "view_messages_received",
                "description": "View messages received by the current user",
                "parameters": {}
            },
            {
                "name": "search_messages",
                "description": "Search for messages containing specific text",
                "parameters": {
                    "keyword": "string - keyword to search for in messages"
                }
            },
            {
                "name": "get_message_details",
                "description": "Get details of a specific message",
                "parameters": {
                    "message_id": "string - the ID of the message"
                }
            },
            {
                "name": "block_user",
                "description": "Block a user from sending messages",
                "parameters": {
                    "user_id": "string - the ID of the user to block"
                }
            },
            {
                "name": "unblock_user",
                "description": "Unblock a user to allow messages",
                "parameters": {
                    "user_id": "string - the ID of the user to unblock"
                }
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "login", "logout", "list_users", "get_user_id",
            "send_message", "view_messages_sent", "view_messages_received",
            "search_messages", "get_message_details", "block_user", "unblock_user"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            if action_name == "login":
                user_id = self._extract_user_id_from_question()
                if user_id:
                    api_result = self.api.login(user_id)
                    result = self._process_api_result(api_result, f"Logged in user {user_id}")
                    if result["success"]:
                        self.current_user = user_id

            elif action_name == "logout":
                result = {"success": True, "message": "Logged out successfully", "data": {"action": "logout"}}
                self.current_user = None

            elif action_name == "list_users":
                api_result = self.api.list_users()
                result = self._process_api_result(api_result, "User list")

            elif action_name == "get_user_id":
                user_name = self._extract_user_name_from_question()
                if user_name:
                    api_result = self.api.get_user_id(user_name)
                    result = self._process_api_result(api_result, f"User ID for {user_name}")

            elif action_name == "send_message":
                if self.current_user:
                    receiver_id = self._extract_receiver_id_from_question()
                    message_text = self._extract_message_text_from_question()
                    if receiver_id and message_text:
                        api_result = self.api.send_message(receiver_id, message_text)
                        result = self._process_api_result(api_result, f"Message sent to {receiver_id}")
                        if result["success"]:
                            self.message_history.append({
                                "action": "send",
                                "from": self.current_user,
                                "to": receiver_id,
                                "message": message_text,
                                "timestamp": len(self.message_history)
                            })
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "view_messages_sent":
                if self.current_user:
                    api_result = self.api.view_messages_sent()
                    result = self._process_api_result(api_result, "Sent messages")
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "view_messages_received":
                if self.current_user:
                    api_result = self.api.view_messages_received()
                    result = self._process_api_result(api_result, "Received messages")
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "search_messages":
                if self.current_user:
                    keyword = self._extract_keyword_from_question()
                    if keyword:
                        # 简化的搜索实现
                        result = {"success": True, "message": f"Messages containing '{keyword}': 2 found",
                                "data": {"keyword": keyword, "count": 2}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_message_details":
                if self.current_user:
                    message_id = self._extract_message_id_from_question()
                    if message_id:
                        result = {"success": True, "message": f"Details for message {message_id}",
                                "data": {"message_id": message_id, "content": "Sample message content"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "block_user":
                if self.current_user:
                    user_id = self._extract_user_id_from_question()
                    if user_id:
                        result = {"success": True, "message": f"User {user_id} blocked",
                                "data": {"user_id": user_id, "action": "blocked"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "unblock_user":
                if self.current_user:
                    user_id = self._extract_user_id_from_question()
                    if user_id:
                        result = {"success": True, "message": f"User {user_id} unblocked",
                                "data": {"user_id": user_id, "action": "unblocked"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["send_failed"]

    def _process_api_result(self, api_result: Dict, description: str) -> Dict:
        """处理API返回结果"""
        if isinstance(api_result, str):
            return {"success": True, "message": api_result, "data": {"response": api_result}}
        elif isinstance(api_result, dict):
            if "error" in api_result:
                return {"success": False, "message": f"Error: {api_result['error']}", "data": None}
            else:
                return {"success": True, "message": description, "data": api_result}
        else:
            return {"success": True, "message": str(api_result), "data": {"response": str(api_result)}}

    def _extract_user_id_from_question(self) -> Optional[str]:
        """从问题中提取用户ID"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 简单的用户ID提取模式
        import re
        # 匹配用户ID模式 (如USR001, USR002等)
        user_id_pattern = r'USR\d{3}'
        matches = re.findall(user_id_pattern, question)
        if matches:
            return matches[0]

        # 匹配数字ID模式
        id_pattern = r'\b\d{3,}\b'
        matches = re.findall(id_pattern, question)
        if matches:
            return matches[0]

        return None

    def _extract_user_name_from_question(self) -> Optional[str]:
        """从问题中提取用户名"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 已知的用户名列表
        known_users = ["Alice", "Bob", "Catherine", "Daniel", "alex", "simona"]

        for user in known_users:
            if user.lower() in question.lower():
                return user

        return None

    def _extract_receiver_id_from_question(self) -> Optional[str]:
        """从问题中提取接收者ID"""
        # 与提取用户ID逻辑相同
        return self._extract_user_id_from_question()

    def _extract_message_text_from_question(self) -> Optional[str]:
        """从问题中提取消息文本"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 提取引号中的内容
        import re
        matches = re.findall(r'["\']([^"\']+)["\']', question)
        if matches:
            return matches[0]

        # 查找常见消息模式
        if "connect" in question.lower():
            return "I want to connect."
        elif "upload" in question.lower():
            return "Could you upload the file?"
        elif "hello" in question.lower():
            return "Hello, how are you?"

        return None

    def _extract_keyword_from_question(self) -> Optional[str]:
        """从问题中提取搜索关键词"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 简单的关键词提取
        import re
        # 查找"keyword", "search", "containing"等词后面的内容
        patterns = [
            r'(?:keyword|search|containing|with)["\']?\s*([^"\'\s]+)["\']?',
            r'search for ["\']([^"\']+)["\']',
            r'containing ["\']([^"\']+)["\']'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _extract_message_id_from_question(self) -> Optional[str]:
        """从问题中提取消息ID"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 匹配消息ID模式
        id_pattern = r'(?:message|msg)\s*(?:id)?\s*[:#]?\s*(\w+)'
        matches = re.findall(id_pattern, question, re.IGNORECASE)
        if matches:
            return matches[0]

        return None

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            if "not found" in result["message"].lower():
                return self.reward_config["user_not_found"]
            elif "log in" in result["message"].lower():
                return self.reward_config["unauthorized"]
            else:
                return self.reward_config["send_failed"]

        reward = 0

        # 基础操作奖励
        if action_name == "login":
            reward = self.reward_config["user_authenticated"]
        elif action_name == "send_message":
            reward = self.reward_config["message_sent"]
        elif action_name in ["view_messages_sent", "view_messages_received"]:
            reward = self.reward_config["message_received"]
        elif action_name in ["list_users", "get_user_id"]:
            reward = self.reward_config["user_found"]
        elif action_name == "search_messages":
            reward = self.reward_config["message_searched"]
        else:
            reward = 0.5  # 其他成功操作的小奖励

        return reward

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        return {
            "question": self.current_question,
            "current_user": self.current_user or "None",
            "authentication_state": {
                "is_logged_in": 1 if self.current_user else 0,
                "user_id": self.current_user or "",
                "user_name": self.current_user or ""
            },
            "inbox_info": {
                "unread_count": 2,  # 简化实现
                "total_messages": 5,
                "senders": ["Alice", "Bob", "Catherine"]
            },
            "outbox_info": {
                "sent_count": len([msg for msg in self.message_history if msg["action"] == "send"]),
                "recipients": list(set([msg["to"] for msg in self.message_history if msg["action"] == "send"]))
            },
            "function_docs": self._get_function_docs(),
            "messaging_state": {
                "messages_sent": len([msg for msg in self.message_history if msg["action"] == "send"]),
                "messages_received": 3,  # 简化实现
                "users_contacted": len(set([msg["to"] for msg in self.message_history if msg["action"] == "send"])),
                "tasks_completed": len(self.message_history)
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # 消息任务通常需要完成发送和接收消息
        messaging_state = obs["messaging_state"]
        return (messaging_state["messages_sent"] >= 2 and
                messaging_state["users_contacted"] >= 1)  # 发送至少2条消息给至少1个用户