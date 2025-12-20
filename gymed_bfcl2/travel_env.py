"""
旅行预订API的Gymnasium环境实现（基于实际TravelAPI源代码）

将BFCL的旅行预订API转换为强化学习环境，代理需要学习
如何处理复杂的旅行预订任务，包括认证、航班预订、信用卡管理、
预算管理等完整的旅行系统功能。
"""

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from base_env import FunctionCallingEnv
from eval_checker.multi_turn_eval.func_source_code.travel_booking import TravelAPI

# 自定义默认奖励函数配置
DEFAULT_REWARD_CONFIG = {
}

class TravelBookingEnv(FunctionCallingEnv):
    """
    旅行预订环境（基于实际TravelAPI）

    代理需要学习如何使用完整的旅行预订系统，包括用户认证、
    信用卡管理、航班查询与预订、预算管理、保险购买等。
    """

    def __init__(
        self,
        test_entry: Dict[str, Any],  # 直接传入测试条目，不再内部创建
        max_turns: int = 15,
        task_type: str = "travel_booking",
        reward_config: Optional[Dict[str, float]] = None
    ):
        """
        初始化旅行预订环境

        Args:
            test_entry: BFCL 测试条目（从 data/ 文件加载）
            max_turns: 最大轮数
            reward_config: 自定义奖励配置（可选）
        """
        self.test_entry = test_entry
        self.max_turns = max_turns
        self.booking_history = []

        # 初始化TravelAPI实例
        self.travel_api = TravelAPI()

        # 调用父类初始化，传递所有参数
        super().__init__(test_entry, max_turns,task_type,reward_config)

        # 直接从 test_entry 加载场景到 TravelAPI
        self._load_scenario_from_test_entry()

        # 设置观察空间
        self._setup_travel_observation_space()

    def _create_travel_test_entry(self, task_type: str, difficulty: str) -> Dict[str, Any]:
        """创建旅行预订测试条目（基于实际API）"""
        function_docs = [
            {
                "name": "authenticate_travel",
                "description": "Authenticate the user with the travel API",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "client_id": {"type": "string", "description": "The client applications client_id supplied by App Management"},
                        "client_secret": {"type": "string", "description": "The client applications client_secret supplied by App Management"},
                        "refresh_token": {"type": "string", "description": "The refresh token obtained from the initial authentication"},
                        "grant_type": {"type": "string", "description": "The grant type of the authentication request. Here are the options: read_write, read, write"},
                        "user_first_name": {"type": "string", "description": "The first name of the user"},
                        "user_last_name": {"type": "string", "description": "The last name of the user"}
                    },
                    "required": ["client_id", "client_secret", "refresh_token", "grant_type", "user_first_name", "user_last_name"]
                }
            },
            {
                "name": "travel_get_login_status",
                "description": "Get the status of the login",
                "parameters": {
                    "type": "dict",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_budget_fiscal_year",
                "description": "Get the budget fiscal year",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "lastModifiedAfter": {"type": "string", "description": "[Optional] Use this field if you only want Fiscal Years that were changed after the supplied date. The supplied date will be interpreted in the UTC time zone. If lastModifiedAfter is not supplied, the service will return all Fiscal Years, regardless of modified date. Example: 2016-03-29T16:12:20. Return in the format of YYYY-MM-DDTHH:MM:SS."},
                        "includeRemoved": {"type": "string", "description": "[Optional] If true, the service will return all Fiscal Years, including those that were previously removed. If not supplied, this field defaults to false."}
                    },
                    "required": []
                }
            },
            {
                "name": "register_credit_card",
                "description": "Register a credit card",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "access_token": {"type": "string", "description": "The access token obtained from the authenticate method"},
                        "card_number": {"type": "string", "description": "The credit card number"},
                        "expiration_date": {"type": "string", "description": "The expiration date of the credit card in the format MM/YYYY"},
                        "cardholder_name": {"type": "string", "description": "The name of the cardholder"},
                        "card_verification_number": {"type": "integer", "description": "The card verification number"}
                    },
                    "required": ["access_token", "card_number", "expiration_date", "cardholder_name", "card_verification_number"]
                }
            },
            {
                "name": "get_flight_cost",
                "description": "Get the list of cost of a flight in USD based on location, date, and class",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "travel_from": {"type": "string", "description": "The 3 letter code of the departing airport"},
                        "travel_to": {"type": "string", "description": "The 3 letter code of the arriving airport"},
                        "travel_date": {"type": "string", "description": "The date of the travel in the format 'YYYY-MM-DD'"},
                        "travel_class": {"type": "string", "description": "The class of the travel. Options are: economy, business, first."}
                    },
                    "required": ["travel_from", "travel_to", "travel_date", "travel_class"]
                }
            },
            {
                "name": "get_credit_card_balance",
                "description": "Get the balance of a credit card",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "access_token": {"type": "string", "description": "The access token obtained from the authenticate"},
                        "card_id": {"type": "string", "description": "The ID of the credit card"}
                    },
                    "required": ["access_token", "card_id"]
                }
            },
            {
                "name": "book_flight",
                "description": "Book a flight given the travel information. From and To should be the airport codes in the IATA format.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "access_token": {"type": "string", "description": "The access token obtained from the authenticate"},
                        "card_id": {"type": "string", "description": "The ID of the credit card to use for the booking"},
                        "travel_date": {"type": "string", "description": "The date of the travel in the format YYYY-MM-DD"},
                        "travel_from": {"type": "string", "description": "The location the travel is from"},
                        "travel_to": {"type": "string", "description": "The location the travel is to"},
                        "travel_class": {"type": "string", "description": "The class of the travel"}
                    },
                    "required": ["access_token", "card_id", "travel_date", "travel_from", "travel_to", "travel_class"]
                }
            },
            {
                "name": "retrieve_invoice",
                "description": "Retrieve the invoice for a booking",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "access_token": {"type": "string", "description": "The access token obtained from the authenticate"},
                        "booking_id": {"type": "string", "description": "[Optional] The ID of the booking"},
                        "insurance_id": {"type": "string", "description": "[Optional] The ID of the insurance"}
                    },
                    "required": ["access_token"]
                }
            },
            {
                "name": "list_all_airports",
                "description": "List all available airports",
                "parameters": {
                    "type": "dict",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "cancel_booking",
                "description": "Cancel a booking",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "access_token": {"type": "string", "description": "The access token obtained from the authenticate"},
                        "booking_id": {"type": "string", "description": "The ID of the booking"}
                    },
                    "required": ["access_token", "booking_id"]
                }
            },
            {
                "name": "compute_exchange_rate",
                "description": "Compute the exchange rate between two currencies",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base_currency": {"type": "string", "description": "The base currency. [Enum]: USD, RMB, EUR, JPY, GBP, CAD, AUD, INR, RUB, BRL, MXN"},
                        "target_currency": {"type": "string", "description": "The target currency. [Enum]: USD, RMB, EUR, JPY, GBP, CAD, AUD, INR, RUB, BRL, MXN"},
                        "value": {"type": "float", "description": "The value to convert"}
                    },
                    "required": ["base_currency", "target_currency", "value"]
                }
            },
            {
                "name": "verify_traveler_information",
                "description": "Verify the traveler information",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "first_name": {"type": "string", "description": "The first name of the traveler"},
                        "last_name": {"type": "string", "description": "The last name of the traveler"},
                        "date_of_birth": {"type": "string", "description": "The date of birth of the traveler in the format YYYY-MM-DD"},
                        "passport_number": {"type": "string", "description": "The passport number of the traveler"}
                    },
                    "required": ["first_name", "last_name", "date_of_birth", "passport_number"]
                }
            },
            {
                "name": "set_budget_limit",
                "description": "Set the budget limit for the user",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "access_token": {"type": "string", "description": "The access token obtained from the authentication process or initial configuration."},
                        "budget_limit": {"type": "float", "description": "The budget limit to set in USD"}
                    },
                    "required": ["access_token", "budget_limit"]
                }
            },
            {
                "name": "get_nearest_airport_by_city",
                "description": "Get the nearest airport to the given location",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string", "description": "The name of the location. [Enum]: Rivermist, Stonebrook, Maplecrest, Silverpine, Shadowridge, London, Paris, Sunset Valley, Oakendale, Willowbend, Crescent Hollow, Autumnville, Pinehaven, Greenfield, San Francisco, Los Angeles, New York, Chicago, Boston, Beijing, Hong Kong, Rome, Tokyo"}
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "purchase_insurance",
                "description": "Purchase insurance",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "access_token": {"type": "string", "description": "The access token obtained from the authenticate"},
                        "insurance_type": {"type": "string", "description": "The type of insurance to purchase"},
                        "booking_id": {"type": "string", "description": "The ID of the booking"},
                        "insurance_cost": {"type": "float", "description": "The cost of the insurance"},
                        "card_id": {"type": "string", "description": "The ID of the credit card to use for the"}
                    },
                    "required": ["access_token", "insurance_type", "booking_id", "insurance_cost", "card_id"]
                }
            },
            {
                "name": "contact_customer_support",
                "description": "Contact travel booking customer support, get immediate support on an issue with an online call.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "booking_id": {"type": "string", "description": "The ID of the booking"},
                        "message": {"type": "string", "description": "The message to send to customer support"}
                    },
                    "required": ["booking_id", "message"]
                }
            },
            {
                "name": "get_all_credit_cards",
                "description": "Get all registered credit cards",
                "parameters": {
                    "type": "dict",
                    "properties": {},
                    "required": []
                }
            }
        ]

        # 直接使用传入的 test_entry，不再生成测试条目
        # question 和其他字段直接从 test_entry 获取
        return self.test_entry



    def _load_scenario(self):
        """加载旅行场景（已弃用，保留向后兼容性）"""
        # 从 test_entry 中获取 initial_config 并加载
        if "initial_config" in self.test_entry:
            self.travel_api._load_scenario(self.test_entry["initial_config"], self.long_context)
        else:
            # 如果没有 initial_config，使用默认配置
            default_scenario = {
                "random_seed": 141053,
                "credit_card_list": {},
                "booking_record": {},
                "access_token": None,
                "token_type": None,
                "token_expires_in": None,
                "token_scope": None,
                "user_first_name": None,
                "user_last_name": None,
                "budget_limit": 8000.0 if self.difficulty_level == "medium" else 5000.0,
            }
            self.travel_api._load_scenario(default_scenario, self.long_context)

    def _load_scenario_from_test_entry(self):
        """从传入的 test_entry 加载场景到 TravelAPI"""
        # 从 test_entry 中获取 initial_config 并加载
        if "initial_config" in self.test_entry and "TravelAPI" in self.test_entry["initial_config"]:
            # 直接传递配置到 TravelAPI
            self.travel_api._load_scenario(self.test_entry["initial_config"]["TravelAPI"])
        else:
            # 如果没有 initial_config，使用默认配置
            default_scenario = {
                "TravelAPI": {
                    "random_seed": 141053,
                    "credit_card_list": {},
                    "booking_record": {},
                    "access_token": None,
                    "token_type": None,
                    "token_expires_in": None,
                    "token_scope": None,
                    "user_first_name": None,
                    "user_last_name": None,
                    "budget_limit": 8000.0 if hasattr(self, 'difficulty_level') and self.difficulty_level == "medium" else 5000.0,
                }
            }
            self.travel_api._load_scenario(default_scenario)

    def _setup_travel_observation_space(self):
        """设置旅行预订环境的观察空间"""
        # 不重新定义观察空间，使用父类的默认设置
        pass

    def _get_observation(self) -> Dict:
        """获取旅行预订环境的观察状态"""
        base_obs = super()._get_observation()
        if "available_actions" not in base_obs:
            # 如果不存在，添加默认的可用动作列表
            action_names = [func["name"] for func in self.test_entry.get("function", [])]
            base_obs["available_actions"] = action_names  # 简单列表，让父类处理
        return base_obs

    def _execute_function_call(self, function_call: str) -> Tuple[str, bool]:
        """执行函数调用并记录操作"""
        function_name = self._extract_function_name(function_call)
        parameters = self._extract_parameters(function_call)

        try:
            # 直接调用TravelAPI的原始方法
            if function_name == "authenticate_travel":
                result = self.travel_api.authenticate_travel(**parameters)
                success = True
            elif function_name == "travel_get_login_status":
                result = self.travel_api.travel_get_login_status()
                success = True
            elif function_name == "get_budget_fiscal_year":
                result = self.travel_api.get_budget_fiscal_year(**parameters)
                success = True
            elif function_name == "register_credit_card":
                result = self.travel_api.register_credit_card(**parameters)
                success = True
            elif function_name == "get_flight_cost":
                result = self.travel_api.get_flight_cost(**parameters)
                success = True
            elif function_name == "get_credit_card_balance":
                result = self.travel_api.get_credit_card_balance(**parameters)
                success = True
            elif function_name == "book_flight":
                result = self.travel_api.book_flight(**parameters)
                success = True
            elif function_name == "retrieve_invoice":
                result = self.travel_api.retrieve_invoice(**parameters)
                success = True
            elif function_name == "list_all_airports":
                result = {"airports": self.travel_api.list_all_airports()}
                success = True
            elif function_name == "cancel_booking":
                result = self.travel_api.cancel_booking(**parameters)
                success = True
            elif function_name == "compute_exchange_rate":
                result = self.travel_api.compute_exchange_rate(**parameters)
                success = True
            elif function_name == "verify_traveler_information":
                result = self.travel_api.verify_traveler_information(**parameters)
                success = True
            elif function_name == "set_budget_limit":
                result = self.travel_api.set_budget_limit(**parameters)
                success = True
            elif function_name == "get_nearest_airport_by_city":
                result = self.travel_api.get_nearest_airport_by_city(**parameters)
                success = True
            elif function_name == "purchase_insurance":
                result = self.travel_api.purchase_insurance(**parameters)
                success = True
            elif function_name == "contact_customer_support":
                result = self.travel_api.contact_customer_support(**parameters)
                success = True
            elif function_name == "get_all_credit_cards":
                result = self.travel_api.get_all_credit_cards()
                success = True
            else:
                result = f"Function {function_name} not found in TravelAPI"
                success = False

        except Exception as e:
            result = f"Error executing {function_name}: {str(e)}"
            success = False

        # 记录操作
        operation = {
            "function": function_name,
            "result": str(result),
            "success": success,
            "parameters": parameters
        }

        self.booking_history.append(operation)

        return str(result), success

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
        """计算旅行预订环境的奖励"""
        reward = super()._compute_reward(execution_success, execution_result)
        # 自定义关于执行函数的reward_config
        # if execution_success and self.booking_history:
        #     last_op = self.booking_history[-1]
        #     function_name = last_op["function"]
        #
        #     # 根据不同的函数类型给予奖励
        #     if function_name == "authenticate_travel":
        #         reward += self.reward_config["successful_authentication"]
        #     elif function_name == "travel_get_login_status":
        #         reward += self.reward_config["login_status_check"]
        #     elif function_name == "get_budget_fiscal_year":
        #         reward += self.reward_config["fiscal_year_query"]
        #     elif function_name == "register_credit_card":
        #         reward += self.reward_config["credit_card_management"]
        #     elif function_name == "get_all_credit_cards":
        #         reward += self.reward_config["credit_card_list_query"]
        #     elif function_name == "get_flight_cost":
        #         reward += self.reward_config["correct_flight_selection"]
        #     elif function_name == "list_all_airports":
        #         reward += self.reward_config["airport_query"]
        #     elif function_name == "get_nearest_airport_by_city":
        #         reward += self.reward_config["airport_query"]
        #     elif function_name == "book_flight":
        #         reward += self.reward_config["successful_booking"]
        #     elif function_name == "retrieve_invoice":
        #         reward += self.reward_config["invoice_retrieval"]
        #     elif function_name == "cancel_booking":
        #         reward += self.reward_config["successful_cancellation"]
        #     elif function_name == "compute_exchange_rate":
        #         reward += self.reward_config["exchange_rate_computation"]
        #     elif function_name == "verify_traveler_information":
        #         reward += self.reward_config["traveler_verification"]
        #     elif function_name == "contact_customer_support":
        #         reward += self.reward_config["customer_support_contact"]
        #     elif function_name == "purchase_insurance":
        #         reward += self.reward_config["insurance_purchase"]
        #     elif function_name == "set_budget_limit":
        #         reward += self.reward_config["budget_compliance"]
        #     elif function_name == "get_credit_card_balance":
        #         reward += self.reward_config["credit_card_management"]
        #
        #     # 操作失败惩罚
        #     if not execution_success:
        #         reward += self.reward_config["failed_operation"]
        #         if "Error" in execution_result and "parameter" in execution_result.lower():
        #             reward += self.reward_config["invalid_parameters"]

        return reward

    def render(self):
        """渲染环境状态"""
        super().render()

        print(f"Authenticated: {'Yes' if self.travel_api.access_token else 'No'}")
        print(f"User: {self.travel_api.user_first_name or 'None'} {self.travel_api.user_last_name or 'None'}")
        print(f"Budget Limit: ${self.travel_api.budget_limit or 0}")
        print(f"Credit Cards: {len(self.travel_api.credit_card_list)}")
        print(f"Bookings: {len(self.travel_api.booking_record)}")
        print(f"Token Expires In: {self.travel_api.token_expires_in or 0}")

        if self.booking_history:
            last_op = self.booking_history[-1]
            print(f"Last Action: {last_op.get('function', 'Unknown')} - {'Success' if last_op.get('success', False) else 'Failed'}")

        print("=" * 60)


# 注册环境以便使用 gym.make() 创建
gym.register(
    id="travel_booking/TravelBookingEnv-v0",
    entry_point=TravelBookingEnv,
    max_episode_steps=50,
    reward_threshold=0.0,
)