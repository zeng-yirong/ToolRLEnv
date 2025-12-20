from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot, DEFAULT_STATE

class TradingBotEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.trading_bot_api = TradingBot()
        self.reward_config = {
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
        self.current_user = None
        self.trading_history = []
        self.portfolio_value = 10000.0  # 初始资金
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        # 从 test_entry 中获取 initial_config 并加载
        if "initial_config" in self.test_entry and "TradingBot" in self.test_entry["initial_config"]:
            # 直接传递配置到 TravelAPI
            self.trading_bot_api._load_scenario(self.test_entry["initial_config"]["TradingBot"])
        else:
            default_scenario = DEFAULT_STATE
            self.trading_bot_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "_generate_transaction_timestamp":
                result = self.trading_bot_api._generate_transaction_timestamp()
                success = True

            elif function_name == "get_current_time":
                result = self.trading_bot_api.get_current_time()
                success = True

            elif function_name == "update_market_status":
                result = self.trading_bot_api.update_market_status(**parameters)
                success = True

            elif function_name == "get_symbol_by_name":
                result = self.trading_bot_api.get_symbol_by_name()
                success = True

            elif function_name == "get_stock_info":
                result = self.trading_bot_api.get_stock_info(**parameters)
                success = True

            elif function_name == "get_order_details":
                result = self.trading_bot_api.get_order_details(**parameters)
                success = True

            elif function_name == "cancel_order":
                result = self.trading_bot_api.cancel_order(**parameters)
                success = True

            elif function_name == "place_order":
                result = self.trading_bot_api.place_order(**parameters)
                success = True

            elif function_name == "make_transaction":
                result = self.trading_bot_api.make_transaction(**parameters)
                success = True

            elif function_name == "get_account_info":
                result = self.trading_bot_api.get_account_info()
                success = True

            elif function_name == "trading_login":
                result = self.trading_bot_api.trading_login(**parameters)
                success = True

            elif function_name == "trading_get_login_status":
                result = self.trading_bot_api.trading_get_login_status()
                success = True

            elif function_name == "trading_logout":
                result = self.trading_bot_api.trading_logout()
                success = True

            elif function_name == "fund_account":
                result = self.trading_bot_api.fund_account(**parameters)
                success = True

            elif function_name == "remove_stock_from_watchlist":
                result = self.trading_bot_api.remove_stock_from_watchlist(**parameters)
                success = True

            elif function_name == "get_watchlist":
                result = self.trading_bot_api.get_watchlist()
                success = True

            elif function_name == "get_order_history":
                result = self.trading_bot_api.get_order_history()
                success = True

            elif function_name == "get_transaction_history":
                result = self.trading_bot_api.get_transaction_history(**parameters)
                success = True

            elif function_name == "update_stock_price":
                result = self.trading_bot_api.update_stock_price(**parameters)
                success = True

            elif function_name == "get_available_stocks":
                result = self.trading_bot_api.get_available_stocks(**parameters)
                success = True

            elif function_name == "filter_stocks_by_price":
                result = self.trading_bot_api.filter_stocks_by_price(**parameters)
                success = True

            elif function_name == "add_to_watchlist":
                result = self.trading_bot_api.add_to_watchlist(**parameters)
                success = True

            elif function_name == "notify_price_change":
                result = self.trading_bot_api.notify_price_change(**parameters)
                success = True

            else:
                result = f"Function {function_name} not found in TradingBot"
                success = False

        except Exception as e:
            result = f"Error executing {function_name}: {str(e)}"
            success = False

        operation = {
            "function": function_name,
            "result": str(result),
            "success": success,
            "parameters": parameters
        }

        self.trading_history.append(operation)

        return str(result), success