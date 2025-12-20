"""
交易机器人环境的Gymnasium适配

基于TradingBot实现的金融交易环境，
支持股票查询、交易下单、投资组合管理等金融操作。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json
from copy import deepcopy

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class TradingBotEnv(FunctionCallingEnv):
    """
    金融交易环境

    支持以下交易操作：
    - 账户管理：login, get_account_info, get_balance
    - 市场数据：get_stock_price, get_market_status, search_stocks
    - 交易操作：buy_stock, sell_stock, place_order, cancel_order
    - 投资组合：get_portfolio, calculate_profit_loss, get_order_history
    - 分析工具：get_stock_info, calculate_returns, analyze_performance
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "financial_trading",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化交易机器人环境

        Args:
            test_entry: 测试条目，包含交易系统初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入TradingBot
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot, DEFAULT_STATE

        self.api = TradingBot()

        # 默认奖励配置
        default_reward_config = {
            "trade_executed": 3.0,          # 成功执行交易
            "profit_made": 5.0,             # 获得利润
            "order_placed": 2.0,            # 成功下单
            "market_analyzed": 1.5,         # 完成市场分析
            "portfolio_updated": 1.0,       # 投资组合更新
            "task_completion": 5.0,         # 任务完成
            "trade_failed": -3.0,           # 交易失败
            "insufficient_funds": -2.5,     # 资金不足
            "market_closed": -2.0,          # 市场关闭
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)

        # 加载交易系统初始状态
        if test_entry and "initial_config" in test_entry:
            if "TradingBot" in test_entry["initial_config"]:
                self.api._load_scenario(test_entry["initial_config"]["TradingBot"])

        # 初始化交易状态
        self.current_user = None
        self.trading_history = []
        self.portfolio_value = 10000.0  # 初始资金

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        actions = [
            "login", "get_account_info", "get_balance", "get_stock_price", "get_market_status",
            "search_stocks", "buy_stock", "sell_stock", "place_order", "cancel_order",
            "get_portfolio", "calculate_profit_loss", "get_order_history", "get_stock_info",
            "calculate_returns", "analyze_performance"
        ]
        return spaces.Discrete(len(actions))

    def _get_observation_space(self) -> spaces.Space:
        """定义观察空间"""
        return spaces.Dict({
            "question": spaces.Text(1000),
            "account_info": spaces.Dict({
                "account_id": spaces.Text(50),
                "balance": spaces.Box(low=0, high=1e9, shape=(1,), dtype=np.float32),
                "available_funds": spaces.Box(low=0, high=1e9, shape=(1,), dtype=np.float32),
                "portfolio_value": spaces.Box(low=0, high=1e9, shape=(1,), dtype=np.float32)
            }),
            "market_status": spaces.Dict({
                "is_open": spaces.Discrete(2),
                "current_time": spaces.Text(50),
                "market_trend": spaces.Text(20)
            }),
            "portfolio": spaces.Sequence(spaces.Dict({
                "symbol": spaces.Text(10),
                "quantity": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.int32),
                "avg_price": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
                "current_price": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
                "value": spaces.Box(low=0, high=1e6, shape=(1,), dtype=np.float32)
            })),
            "recent_trades": spaces.Sequence(spaces.Dict({
                "symbol": spaces.Text(10),
                "action": spaces.Text(10),
                "quantity": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
                "price": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
                "timestamp": spaces.Text(50)
            })),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "trading_state": spaces.Dict({
                "trades_executed": spaces.Discrete(100),
                "total_profit_loss": spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32),
                "win_rate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "risk_level": spaces.Discrete(5)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "login",
                "description": "Log in to the trading platform",
                "parameters": {
                    "account_id": "string - trading account ID",
                    "password": "string - account password"
                }
            },
            {
                "name": "get_account_info",
                "description": "Get detailed account information",
                "parameters": {}
            },
            {
                "name": "get_balance",
                "description": "Get current account balance",
                "parameters": {}
            },
            {
                "name": "get_stock_price",
                "description": "Get current stock price",
                "parameters": {
                    "symbol": "string - stock symbol (e.g., AAPL)"
                }
            },
            {
                "name": "get_market_status",
                "description": "Get current market status",
                "parameters": {}
            },
            {
                "name": "search_stocks",
                "description": "Search for stocks by name or symbol",
                "parameters": {
                    "query": "string - search query"
                }
            },
            {
                "name": "buy_stock",
                "description": "Buy stocks",
                "parameters": {
                    "symbol": "string - stock symbol",
                    "quantity": "int - number of shares to buy",
                    "order_type": "optional string - order type (market/limit)"
                }
            },
            {
                "name": "sell_stock",
                "description": "Sell stocks",
                "parameters": {
                    "symbol": "string - stock symbol",
                    "quantity": "int - number of shares to sell",
                    "order_type": "optional string - order type (market/limit)"
                }
            },
            {
                "name": "place_order",
                "description": "Place a trading order",
                "parameters": {
                    "symbol": "string - stock symbol",
                    "order_type": "string - buy/sell",
                    "quantity": "int - number of shares",
                    "price": "optional float - limit price"
                }
            },
            {
                "name": "cancel_order",
                "description": "Cancel a pending order",
                "parameters": {
                    "order_id": "string - ID of the order to cancel"
                }
            },
            {
                "name": "get_portfolio",
                "description": "Get current portfolio holdings",
                "parameters": {}
            },
            {
                "name": "calculate_profit_loss",
                "description": "Calculate profit/loss for positions",
                "parameters": {}
            },
            {
                "name": "get_order_history",
                "description": "Get order history",
                "parameters": {}
            },
            {
                "name": "get_stock_info",
                "description": "Get detailed stock information",
                "parameters": {
                    "symbol": "string - stock symbol"
                }
            },
            {
                "name": "calculate_returns",
                "description": "Calculate investment returns",
                "parameters": {
                    "period": "optional string - time period for calculation"
                }
            },
            {
                "name": "analyze_performance",
                "description": "Analyze trading performance",
                "parameters": {}
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "login", "get_account_info", "get_balance", "get_stock_price", "get_market_status",
            "search_stocks", "buy_stock", "sell_stock", "place_order", "cancel_order",
            "get_portfolio", "calculate_profit_loss", "get_order_history", "get_stock_info",
            "calculate_returns", "analyze_performance"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            if action_name == "login":
                account_id = self._extract_account_id_from_question()
                if account_id:
                    self.current_user = account_id
                    result = {"success": True, "message": f"Successfully logged into account {account_id}",
                            "data": {"account_id": account_id, "authenticated": True}}
                else:
                    result = {"success": False, "message": "Please provide a valid account ID", "data": None}

            elif action_name == "get_account_info":
                if self.current_user:
                    result = {"success": True, "message": "Account information retrieved",
                            "data": {
                                "account_id": self.current_user,
                                "account_type": "margin",
                                "created_date": "2020-01-01",
                                "status": "active"
                            }}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_balance":
                if self.current_user:
                    result = {"success": True, "message": f"Current balance: ${self.portfolio_value:.2f}",
                            "data": {"balance": self.portfolio_value}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_stock_price":
                symbol = self._extract_stock_symbol_from_question()
                if symbol:
                    # 简化的价格获取
                    price_map = {"AAPL": 175.50, "GOOGL": 2850.25, "MSFT": 380.75, "TSLA": 250.30}
                    price = price_map.get(symbol.upper(), 100.0)
                    result = {"success": True, "message": f"Current price of {symbol}: ${price:.2f}",
                            "data": {"symbol": symbol, "price": price}}
                else:
                    result = {"success": False, "message": "Please provide a valid stock symbol", "data": None}

            elif action_name == "get_market_status":
                # 简化的市场状态
                result = {"success": True, "message": "Market is currently open",
                        "data": {
                            "status": "open",
                            "current_time": "10:30 AM EST",
                            "market_trend": "bullish"
                        }}

            elif action_name == "search_stocks":
                query = self._extract_search_query_from_question()
                if query:
                    result = {"success": True, "message": f"Found stocks matching '{query}'",
                            "data": {
                                "query": query,
                                "results": [
                                    {"symbol": "AAPL", "name": "Apple Inc.", "price": 175.50},
                                    {"symbol": "MSFT", "name": "Microsoft Corporation", "price": 380.75}
                                ]
                            }}
                else:
                    result = {"success": False, "message": "Please provide a search query", "data": None}

            elif action_name == "buy_stock":
                if self.current_user:
                    symbol = self._extract_stock_symbol_from_question()
                    quantity = self._extract_quantity_from_question()

                    if symbol and quantity:
                        price_map = {"AAPL": 175.50, "GOOGL": 2850.25, "MSFT": 380.75, "TSLA": 250.30}
                        price = price_map.get(symbol.upper(), 100.0)
                        total_cost = price * quantity

                        if total_cost <= self.portfolio_value:
                            self.portfolio_value -= total_cost
                            self.trading_history.append({
                                "action": "buy",
                                "symbol": symbol,
                                "quantity": quantity,
                                "price": price,
                                "total": total_cost,
                                "timestamp": len(self.trading_history)
                            })
                            result = {"success": True, "message": f"Bought {quantity} shares of {symbol} at ${price:.2f}",
                                    "data": {"symbol": symbol, "quantity": quantity, "price": price, "total": total_cost}}
                        else:
                            result = {"success": False, "message": "Insufficient funds", "data": None}
                    else:
                        result = {"success": False, "message": "Please provide symbol and quantity", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "sell_stock":
                if self.current_user:
                    symbol = self._extract_stock_symbol_from_question()
                    quantity = self._extract_quantity_from_question()

                    if symbol and quantity:
                        price_map = {"AAPL": 175.50, "GOOGL": 2850.25, "MSFT": 380.75, "TSLA": 250.30}
                        price = price_map.get(symbol.upper(), 100.0)
                        total_value = price * quantity

                        self.portfolio_value += total_value
                        self.trading_history.append({
                            "action": "sell",
                            "symbol": symbol,
                            "quantity": quantity,
                            "price": price,
                            "total": total_value,
                            "timestamp": len(self.trading_history)
                        })
                        result = {"success": True, "message": f"Sold {quantity} shares of {symbol} at ${price:.2f}",
                                "data": {"symbol": symbol, "quantity": quantity, "price": price, "total": total_value}}
                    else:
                        result = {"success": False, "message": "Please provide symbol and quantity", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "place_order":
                if self.current_user:
                    symbol = self._extract_stock_symbol_from_question()
                    order_type = self._extract_order_type_from_question()
                    quantity = self._extract_quantity_from_question()
                    price = self._extract_price_from_question()

                    if symbol and order_type and quantity:
                        order_id = f"ORDER-{len(self.trading_history) + 1001}"
                        result = {"success": True, "message": f"Order {order_id} placed successfully",
                                "data": {
                                    "order_id": order_id,
                                    "symbol": symbol,
                                    "order_type": order_type,
                                    "quantity": quantity,
                                    "price": price
                                }}
                    else:
                        result = {"success": False, "message": "Please provide order details", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_portfolio":
                if self.current_user:
                    # 简化的投资组合
                    portfolio = [
                        {"symbol": "AAPL", "quantity": 10, "avg_price": 170.00, "current_price": 175.50, "value": 1755.00},
                        {"symbol": "MSFT", "quantity": 5, "avg_price": 375.00, "current_price": 380.75, "value": 1903.75}
                    ]
                    result = {"success": True, "message": "Portfolio retrieved",
                            "data": {"portfolio": portfolio, "total_value": self.portfolio_value}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "calculate_profit_loss":
                if self.current_user:
                    # 简化的盈亏计算
                    total_pnl = 1250.75  # 示例值
                    result = {"success": True, "message": f"Total P&L: ${total_pnl:.2f}",
                            "data": {"total_profit_loss": total_pnl}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            else:
                # 其他查询操作的简化实现
                result = {"success": True, "message": f"{action_name} completed successfully", "data": {"action": action_name}}

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["trade_failed"]

    def _extract_account_id_from_question(self) -> Optional[str]:
        """从问题中提取账户ID"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 匹配账户ID模式
        account_pattern = r'\b(ACC\d{6}|\d{6,})\b'
        matches = re.findall(account_pattern, question)
        if matches:
            return matches[0]

        return "123456"  # 默认账户ID

    def _extract_stock_symbol_from_question(self) -> Optional[str]:
        """从问题中提取股票代码"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 常见股票代码
        known_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]

        import re
        # 查找股票代码模式
        symbol_pattern = r'\b[A-Z]{2,5}\b'
        matches = re.findall(symbol_pattern, question)
        if matches:
            for match in matches:
                if match in known_symbols:
                    return match

        # 查找公司名映射
        company_map = {
            "apple": "AAPL",
            "google": "GOOGL",
            "microsoft": "MSFT",
            "tesla": "TSLA",
            "amazon": "AMZN",
            "meta": "META",
            "nvidia": "NVDA"
        }

        for company, symbol in company_map.items():
            if company in question.lower():
                return symbol

        return None

    def _extract_quantity_from_question(self) -> Optional[int]:
        """从问题中提取数量"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 查找数量模式
        quantity_patterns = [
            r'(\d+)\s+shares?',
            r'(\d+)\s+stocks?',
            r'quantity[:\s]+(\d+)',
            r'(\d+)\s+units?'
        ]

        for pattern in quantity_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return int(matches[0])

        return None

    def _extract_order_type_from_question(self) -> Optional[str]:
        """从问题中提取订单类型"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        if "buy" in question.lower():
            return "buy"
        elif "sell" in question.lower():
            return "sell"

        return None

    def _extract_price_from_question(self) -> Optional[float]:
        """从问题中提取价格"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 查找价格模式
        price_patterns = [
            r'\$(\d+\.?\d*)',
            r'price[:\s]+(\d+\.?\d*)',
            r'at\s+(\d+\.?\d*)'
        ]

        for pattern in price_patterns:
            matches = re.findall(pattern, question)
            if matches:
                return float(matches[0])

        return None

    def _extract_search_query_from_question(self) -> Optional[str]:
        """从问题中提取搜索查询"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 查找搜索关键词模式
        patterns = [
            r'search for ["\']([^"\']+)["\']',
            r'search["\']?\s+(?:for\s+)?["\']?([^"\'\s]+)["\']?',
            r'looking for ["\']([^"\']+)["\']'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            if "funds" in result["message"].lower():
                return self.reward_config["insufficient_funds"]
            elif "market" in result["message"].lower() and "closed" in result["message"].lower():
                return self.reward_config["market_closed"]
            else:
                return self.reward_config["trade_failed"]

        # 基础操作奖励
        reward_map = {
            "buy_stock": self.reward_config["trade_executed"],
            "sell_stock": self.reward_config["trade_executed"],
            "place_order": self.reward_config["order_placed"],
            "get_stock_price": self.reward_config["market_analyzed"],
            "search_stocks": self.reward_config["market_analyzed"],
            "get_portfolio": self.reward_config["portfolio_updated"],
            "calculate_profit_loss": self.reward_config["portfolio_updated"],
            "login": 1.0,
            "get_account_info": 0.5,
            "get_balance": 0.5,
            "get_market_status": 0.5
        }

        return reward_map.get(action_name, 0.5)

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        # 简化的投资组合
        portfolio = [
            {
                "symbol": "AAPL",
                "quantity": np.array([10], dtype=np.int32),
                "avg_price": np.array([170.00], dtype=np.float32),
                "current_price": np.array([175.50], dtype=np.float32),
                "value": np.array([1755.00], dtype=np.float32)
            }
        ]

        # 简化的最近交易
        recent_trades = [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": np.array([10], dtype=np.int32),
                "price": np.array([175.50], dtype=np.float32),
                "timestamp": "2024-01-01 10:30:00"
            }
        ]

        return {
            "question": self.current_question,
            "account_info": {
                "account_id": self.current_user or "123456",
                "balance": np.array([self.portfolio_value], dtype=np.float32),
                "available_funds": np.array([self.portfolio_value], dtype=np.float32),
                "portfolio_value": np.array([self.portfolio_value], dtype=np.float32)
            },
            "market_status": {
                "is_open": 1,
                "current_time": "10:30 AM EST",
                "market_trend": "bullish"
            },
            "portfolio": portfolio,
            "recent_trades": recent_trades,
            "function_docs": self._get_function_docs(),
            "trading_state": {
                "trades_executed": len(self.trading_history),
                "total_profit_loss": np.array([1250.75], dtype=np.float32),
                "win_rate": np.array([0.65], dtype=np.float32),
                "risk_level": 2
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # 交易任务通常需要完成一定数量的交易操作
        trading_state = obs["trading_state"]
        return trading_state["trades_executed"] >= 2  # 至少完成2笔交易