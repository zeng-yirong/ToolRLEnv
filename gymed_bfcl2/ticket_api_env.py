"""
工单API环境的Gymnasium适配

基于TicketAPI实现的工单系统环境，
支持工单创建、查询、更新、关闭等客服工单管理功能。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json
from copy import deepcopy

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class TicketAPIEnv(FunctionCallingEnv):
    """
    工单系统环境

    支持以下工单操作：
    - 工单管理：create_ticket, get_ticket, update_ticket, close_ticket
    - 工单查询：list_tickets, search_tickets, get_ticket_status
    - 工单操作：assign_ticket, add_comment, change_priority
    - 用户管理：login, logout, get_user_info
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "ticket_management",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化工单API环境

        Args:
            test_entry: 测试条目，包含工单系统初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入TicketAPI
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.ticket_api import TicketAPI, DEFAULT_STATE

        self.api = TicketAPI()

        # 默认奖励配置
        default_reward_config = {
            "ticket_created": 2.5,          # 成功创建工单
            "ticket_updated": 1.5,          # 成功更新工单
            "ticket_closed": 3.0,           # 成功关闭工单
            "ticket_found": 1.0,            # 找到工单
            "comment_added": 1.0,           # 成功添加评论
            "priority_changed": 1.5,        # 成功修改优先级
            "task_completion": 5.0,         # 任务完成
            "create_failed": -2.0,          # 创建失败
            "ticket_not_found": -1.5,       # 工单未找到
            "unauthorized": -2.0,           # 未授权操作
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)

        # 加载工单系统初始状态
        if test_entry and "initial_config" in test_entry:
            if "TicketAPI" in test_entry["initial_config"]:
                self.api._load_scenario(test_entry["initial_config"]["TicketAPI"])

        # 初始化工单状态
        self.current_user = None
        self.ticket_actions = []

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        # 基于TicketAPI的实际函数
        actions = [
            "login", "logout", "create_ticket", "get_ticket", "update_ticket",
            "close_ticket", "list_tickets", "search_tickets", "get_ticket_status",
            "assign_ticket", "add_comment", "change_priority"
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
                "role": spaces.Text(50),
                "department": spaces.Text(100)
            }),
            "ticket_queue": spaces.Sequence(spaces.Dict({
                "ticket_id": spaces.Text(50),
                "title": spaces.Text(200),
                "status": spaces.Text(50),
                "priority": spaces.Discrete(5),
                "assigned_to": spaces.Text(100),
                "created_time": spaces.Text(50)
            })),
            "ticket_stats": spaces.Dict({
                "total_tickets": spaces.Discrete(1000),
                "open_tickets": spaces.Discrete(100),
                "closed_tickets": spaces.Discrete(500),
                "my_tickets": spaces.Discrete(50)
            }),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "ticket_state": spaces.Dict({
                "tickets_created": spaces.Discrete(20),
                "tickets_updated": spaces.Discrete(30),
                "tickets_closed": spaces.Discrete(10),
                "comments_added": spaces.Discrete(50),
                "priority_changes": spaces.Discrete(15)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "login",
                "description": "Log in to the ticket system",
                "parameters": {
                    "user_id": "string - user ID for login",
                    "password": "string - password for authentication"
                }
            },
            {
                "name": "logout",
                "description": "Log out from the ticket system",
                "parameters": {}
            },
            {
                "name": "create_ticket",
                "description": "Create a new support ticket",
                "parameters": {
                    "title": "string - ticket title",
                    "description": "string - detailed description of the issue",
                    "priority": "optional int - priority level (1-5)"
                }
            },
            {
                "name": "get_ticket",
                "description": "Get details of a specific ticket",
                "parameters": {
                    "ticket_id": "string - ID of the ticket"
                }
            },
            {
                "name": "update_ticket",
                "description": "Update ticket information",
                "parameters": {
                    "ticket_id": "string - ID of the ticket",
                    "title": "optional string - new title",
                    "description": "optional string - new description"
                }
            },
            {
                "name": "close_ticket",
                "description": "Close a ticket",
                "parameters": {
                    "ticket_id": "string - ID of the ticket to close"
                }
            },
            {
                "name": "list_tickets",
                "description": "List all tickets in the system",
                "parameters": {
                    "status": "optional string - filter by status (open/closed/all)",
                    "priority": "optional int - filter by priority"
                }
            },
            {
                "name": "search_tickets",
                "description": "Search for tickets with keywords",
                "parameters": {
                    "query": "string - search query",
                    "status": "optional string - filter by status"
                }
            },
            {
                "name": "get_ticket_status",
                "description": "Get the status of a ticket",
                "parameters": {
                    "ticket_id": "string - ID of the ticket"
                }
            },
            {
                "name": "assign_ticket",
                "description": "Assign a ticket to a user",
                "parameters": {
                    "ticket_id": "string - ID of the ticket",
                    "assignee": "string - user ID to assign to"
                }
            },
            {
                "name": "add_comment",
                "description": "Add a comment to a ticket",
                "parameters": {
                    "ticket_id": "string - ID of the ticket",
                    "comment": "string - comment content"
                }
            },
            {
                "name": "change_priority",
                "description": "Change the priority of a ticket",
                "parameters": {
                    "ticket_id": "string - ID of the ticket",
                    "priority": "int - new priority level (1-5)"
                }
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "login", "logout", "create_ticket", "get_ticket", "update_ticket",
            "close_ticket", "list_tickets", "search_tickets", "get_ticket_status",
            "assign_ticket", "add_comment", "change_priority"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            if action_name == "login":
                user_id = self._extract_user_id_from_question()
                if user_id:
                    self.current_user = user_id
                    result = {"success": True, "message": f"Successfully logged in as {user_id}",
                            "data": {"user_id": user_id, "role": "support_agent"}}
                else:
                    result = {"success": False, "message": "Please provide a valid user ID", "data": None}

            elif action_name == "logout":
                result = {"success": True, "message": "Successfully logged out", "data": {"action": "logout"}}
                self.current_user = None

            elif action_name == "create_ticket":
                if self.current_user:
                    title = self._extract_ticket_title_from_question()
                    description = self._extract_ticket_description_from_question()
                    priority = self._extract_priority_from_question()

                    if title and description:
                        ticket_id = f"TICKET-{len(self.ticket_actions) + 1001}"
                        self.ticket_actions.append({
                            "action": "create",
                            "ticket_id": ticket_id,
                            "title": title,
                            "description": description,
                            "priority": priority,
                            "created_by": self.current_user,
                            "timestamp": len(self.ticket_actions)
                        })
                        result = {"success": True, "message": f"Ticket {ticket_id} created successfully",
                                "data": {"ticket_id": ticket_id, "title": title, "priority": priority}}
                    else:
                        result = {"success": False, "message": "Please provide title and description", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_ticket":
                ticket_id = self._extract_ticket_id_from_question()
                if ticket_id:
                    # 简化的工单获取实现
                    sample_ticket = {
                        "ticket_id": ticket_id,
                        "title": f"Sample ticket {ticket_id}",
                        "description": "This is a sample ticket description",
                        "status": "open",
                        "priority": 3,
                        "assigned_to": "agent1",
                        "created_by": "user1",
                        "created_time": "2024-01-01 10:00:00",
                        "comments": [
                            {"author": "agent1", "content": "Working on this issue", "time": "2024-01-01 11:00:00"}
                        ]
                    }
                    result = {"success": True, "message": f"Ticket details for {ticket_id}", "data": sample_ticket}
                else:
                    result = {"success": False, "message": "Please provide a valid ticket ID", "data": None}

            elif action_name == "update_ticket":
                if self.current_user:
                    ticket_id = self._extract_ticket_id_from_question()
                    title = self._extract_ticket_title_from_question()
                    description = self._extract_ticket_description_from_question()

                    if ticket_id and (title or description):
                        self.ticket_actions.append({
                            "action": "update",
                            "ticket_id": ticket_id,
                            "updated_by": self.current_user,
                            "timestamp": len(self.ticket_actions)
                        })
                        result = {"success": True, "message": f"Ticket {ticket_id} updated successfully",
                                "data": {"ticket_id": ticket_id, "action": "updated"}}
                    else:
                        result = {"success": False, "message": "Please provide ticket ID and updates", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "close_ticket":
                if self.current_user:
                    ticket_id = self._extract_ticket_id_from_question()
                    if ticket_id:
                        self.ticket_actions.append({
                            "action": "close",
                            "ticket_id": ticket_id,
                            "closed_by": self.current_user,
                            "timestamp": len(self.ticket_actions)
                        })
                        result = {"success": True, "message": f"Ticket {ticket_id} closed successfully",
                                "data": {"ticket_id": ticket_id, "action": "closed"}}
                    else:
                        result = {"success": False, "message": "Please provide a valid ticket ID", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "list_tickets":
                if self.current_user:
                    status = self._extract_status_from_question()
                    # 简化的工单列表实现
                    sample_tickets = [
                        {"ticket_id": "TICKET-1001", "title": "Login issue", "status": "open", "priority": 2},
                        {"ticket_id": "TICKET-1002", "title": "Payment problem", "status": "closed", "priority": 4},
                        {"ticket_id": "TICKET-1003", "title": "Feature request", "status": "open", "priority": 1}
                    ]

                    if status:
                        filtered_tickets = [t for t in sample_tickets if t["status"] == status]
                    else:
                        filtered_tickets = sample_tickets

                    result = {"success": True, "message": f"Found {len(filtered_tickets)} tickets",
                            "data": {"tickets": filtered_tickets, "count": len(filtered_tickets)}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "search_tickets":
                if self.current_user:
                    query = self._extract_search_query_from_question()
                    if query:
                        # 简化的搜索实现
                        results = [
                            {"ticket_id": "TICKET-1004", "title": f"Ticket about {query}", "status": "open"}
                        ]
                        result = {"success": True, "message": f"Found {len(results)} tickets matching '{query}'",
                                "data": {"query": query, "results": results}}
                    else:
                        result = {"success": False, "message": "Please provide a search query", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_ticket_status":
                ticket_id = self._extract_ticket_id_from_question()
                if ticket_id:
                    result = {"success": True, "message": f"Ticket {ticket_id} status: open",
                            "data": {"ticket_id": ticket_id, "status": "open"}}
                else:
                    result = {"success": False, "message": "Please provide a valid ticket ID", "data": None}

            elif action_name == "assign_ticket":
                if self.current_user:
                    ticket_id = self._extract_ticket_id_from_question()
                    assignee = self._extract_assignee_from_question()
                    if ticket_id and assignee:
                        result = {"success": True, "message": f"Ticket {ticket_id} assigned to {assignee}",
                                "data": {"ticket_id": ticket_id, "assignee": assignee}}
                    else:
                        result = {"success": False, "message": "Please provide ticket ID and assignee", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "add_comment":
                if self.current_user:
                    ticket_id = self._extract_ticket_id_from_question()
                    comment = self._extract_comment_from_question()
                    if ticket_id and comment:
                        result = {"success": True, "message": f"Comment added to ticket {ticket_id}",
                                "data": {"ticket_id": ticket_id, "comment": comment, "author": self.current_user}}
                    else:
                        result = {"success": False, "message": "Please provide ticket ID and comment", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "change_priority":
                if self.current_user:
                    ticket_id = self._extract_ticket_id_from_question()
                    priority = self._extract_priority_from_question()
                    if ticket_id and priority:
                        result = {"success": True, "message": f"Priority for ticket {ticket_id} changed to {priority}",
                                "data": {"ticket_id": ticket_id, "priority": priority}}
                    else:
                        result = {"success": False, "message": "Please provide ticket ID and priority", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["create_failed"]

    def _extract_user_id_from_question(self) -> Optional[str]:
        """从问题中提取用户ID"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 匹配用户ID模式
        user_id_pattern = r'\b(USR\d{3}|\d{3,})\b'
        matches = re.findall(user_id_pattern, question)
        if matches:
            return matches[0]

        return None

    def _extract_ticket_id_from_question(self) -> Optional[str]:
        """从问题中提取工单ID"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 匹配工单ID模式
        patterns = [
            r'TICKET-\d+',
            r'ticket["\']?\s*(?:id)?\s*[:#]?\s*(\w+)',
            r'(\d{4})',  # 4位数字
            r't["\']?\s*(\d+)'  # t后跟数字
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                if "TICKET-" in pattern:
                    return matches[0]
                else:
                    return f"TICKET-{matches[0]}"

        return None

    def _extract_ticket_title_from_question(self) -> Optional[str]:
        """从问题中提取工单标题"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 提取引号中的内容
        import re
        matches = re.findall(r'["\']([^"\']+)["\']', question)
        if matches:
            return matches[0]

        # 查找常见工单模式
        if "ticket" in question.lower() and ("create" in question.lower() or "new" in question.lower()):
            # 简单的标题提取
            if "login" in question.lower():
                return "Login Issue"
            elif "payment" in question.lower():
                return "Payment Problem"
            elif "feature" in question.lower():
                return "Feature Request"
            elif "bug" in question.lower():
                return "Bug Report"

        return None

    def _extract_ticket_description_from_question(self) -> Optional[str]:
        """从问题中提取工单描述"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 简化的描述提取
        if "cannot" in question.lower() or "unable" in question.lower():
            return "User is unable to complete a task"
        elif "error" in question.lower():
            return "System error occurred"
        elif "request" in question.lower():
            return "User has a request for new functionality"
        elif "issue" in question.lower():
            return "User reported an issue with the system"

        return "General support request"

    def _extract_priority_from_question(self) -> Optional[int]:
        """从问题中提取优先级"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 优先级关键词映射
        priority_map = {
            "urgent": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "lowest": 1
        }

        for keyword, priority in priority_map.items():
            if keyword in question.lower():
                return priority

        # 查找数字
        import re
        numbers = re.findall(r'\b[1-5]\b', question)
        if numbers:
            return int(numbers[0])

        return 3  # 默认中等优先级

    def _extract_status_from_question(self) -> Optional[str]:
        """从问题中提取状态"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        if "open" in question.lower():
            return "open"
        elif "closed" in question.lower() or "close" in question.lower():
            return "closed"
        elif "all" in question.lower():
            return "all"

        return None

    def _extract_search_query_from_question(self) -> Optional[str]:
        """从问题中提取搜索查询"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 查找搜索关键词模式
        patterns = [
            r'search for ["\']([^"\']+)["\']',
            r'search["\']?\s+(?:for\s+)?["\']?([^"\'\s]+)["\']?',
            r'about["\']?\s+["\']?([^"\'\s]+)["\']?'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _extract_assignee_from_question(self) -> Optional[str]:
        """从问题中提取被分配者"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 常见的用户名
        known_users = ["agent1", "agent2", "john", "alice", "bob"]

        for user in known_users:
            if user in question.lower():
                return user

        return None

    def _extract_comment_from_question(self) -> Optional[str]:
        """从问题中提取评论"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 提取引号中的内容
        import re
        matches = re.findall(r'["\']([^"\']+)["\']', question)
        if matches:
            return matches[0]

        return None

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            if "log in" in result["message"].lower():
                return self.reward_config["unauthorized"]
            elif "not found" in result["message"].lower():
                return self.reward_config["ticket_not_found"]
            else:
                return self.reward_config["create_failed"]

        # 基础操作奖励
        reward_map = {
            "create_ticket": self.reward_config["ticket_created"],
            "update_ticket": self.reward_config["ticket_updated"],
            "close_ticket": self.reward_config["ticket_closed"],
            "get_ticket": self.reward_config["ticket_found"],
            "search_tickets": self.reward_config["ticket_found"],
            "add_comment": self.reward_config["comment_added"],
            "change_priority": self.reward_config["priority_changed"],
            "assign_ticket": 1.0,
            "list_tickets": 0.5,
            "get_ticket_status": 0.5,
            "login": 1.0
        }

        return reward_map.get(action_name, 0.5)

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        # 简化的工单队列
        sample_tickets = [
            {
                "ticket_id": "TICKET-1001",
                "title": "Login Issue",
                "status": "open",
                "priority": 2,
                "assigned_to": "agent1",
                "created_time": "2024-01-01 10:00:00"
            },
            {
                "ticket_id": "TICKET-1002",
                "title": "Payment Problem",
                "status": "open",
                "priority": 4,
                "assigned_to": "agent2",
                "created_time": "2024-01-01 11:30:00"
            }
        ]

        return {
            "question": self.current_question,
            "current_user": self.current_user or "None",
            "authentication_state": {
                "is_logged_in": 1 if self.current_user else 0,
                "user_id": self.current_user or "",
                "role": "support_agent" if self.current_user else "",
                "department": "IT Support"
            },
            "ticket_queue": sample_tickets,
            "ticket_stats": {
                "total_tickets": 15,
                "open_tickets": 8,
                "closed_tickets": 7,
                "my_tickets": len([a for a in self.ticket_actions if a.get("created_by") == self.current_user])
            },
            "function_docs": self._get_function_docs(),
            "ticket_state": {
                "tickets_created": len([a for a in self.ticket_actions if a["action"] == "create"]),
                "tickets_updated": len([a for a in self.ticket_actions if a["action"] == "update"]),
                "tickets_closed": len([a for a in self.ticket_actions if a["action"] == "close"]),
                "comments_added": len([a for a in self.ticket_actions if a["action"] == "comment"]),
                "priority_changes": len([a for a in self.ticket_actions if a["action"] == "priority"])
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # 工单任务通常需要完成一定数量的工单操作
        ticket_state = obs["ticket_state"]
        total_actions = (ticket_state["tickets_created"] +
                        ticket_state["tickets_updated"] +
                        ticket_state["tickets_closed"] +
                        ticket_state["comments_added"] +
                        ticket_state["priority_changes"])
        return total_actions >= 3  # 至少完成3个工单操作