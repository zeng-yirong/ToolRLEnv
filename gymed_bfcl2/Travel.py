from typing import Tuple
from typing import Dict, List, Tuple, Any, Optional

from eval_checker.multi_turn_eval.func_source_code.travel_booking import TravelAPI

class TravelBookingEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.booking_history = []
        self.test_entry = test_entry
        self.travel_api = TravelAPI()
        self.reward_config = {}
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        """从传入的 test_entry 加载场景到 TravelAPI"""
        # 从 test_entry 中获取 initial_config 并加载
        if "initial_config" in self.test_entry and "TravelAPI" in self.test_entry["initial_config"]:
            # 直接传递配置到 TravelAPI
            self.travel_api._load_scenario(self.test_entry["initial_config"]["TravelAPI"])
        else:
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

    def execute_function_call(self, function_name,parameters) -> Tuple[str, bool]:
        """执行函数调用并记录操作"""
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

        operation = {
            "function": function_name,
            "result": str(result),
            "success": success,
            "parameters": parameters
        }

        self.booking_history.append(operation)

        return str(result), success