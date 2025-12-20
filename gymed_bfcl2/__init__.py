"""
Gymnasium适配版本的BFCL多轮函数调用环境（基于实际API源代码）

本包将Berkeley Function Calling Leaderboard的多轮评估系统
适配到Gymnasium强化学习环境中，特别基于实际的TravelAPI源代码
进行重构，使函数调用任务可以通过强化学习方法进行训练和评估。

主要特性：
- 基于实际TravelAPI源代码的完整函数调用环境
- 真实的认证、支付、预订系统
- 17个完整的函数API
- 支持长上下文模式
"""

from .base_env import FunctionCallingEnv
from .travel_env import TravelBookingEnv
from .gorilla_file_system_env import GorillaFileSystemEnv
from .math_api_env import MathAPIEnv
from .message_api_env import MessageAPIEnv
from .twitter_api_env import TwitterAPIEnv
from .ticket_api_env import TicketAPIEnv
from .trading_bot_env import TradingBotEnv
from .vehicle_control_env import VehicleControlEnv
from .web_search_env import WebSearchEnv
from .utils import FunctionCallExecutor, StateManager

__all__ = [
    "FunctionCallingEnv",
    "TravelBookingEnv",
    "GorillaFileSystemEnv",
    "MathAPIEnv",
    "MessageAPIEnv",
    "TwitterAPIEnv",
    "TicketAPIEnv",
    "TradingBotEnv",
    "VehicleControlEnv",
    "WebSearchEnv",
    "FunctionCallExecutor",
    "StateManager"
]

__version__ = "2.0.0"
__author__ = "BFCL Gymnasium Adaptation Team"

# 版本说明
version_info = {
    "version": "2.0.0",
    "description": "基于实际TravelAPI源代码的重构版本",
    "key_features": [
        "完整的TravelAPI集成",
        "17个真实函数API",
        "OAuth认证系统",
        "真实的机场和航线数据",
        "信用卡和预算管理",
        "长上下文支持"
    ],
    "api_source": "eval_checker/multi_turn_eval/func_source_code/travel_booking.py",
    "doc_source": "data/multi_turn_func_doc/travel_booking.json"
}