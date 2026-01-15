from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.ticket_api import TicketAPI, DEFAULT_STATE

class TicketEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.ticket_api = TicketAPI()
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        if "initial_config" in self.test_entry and "TicketAPI" in self.test_entry["initial_config"]:
            self.ticket_api._load_scenario(self.test_entry["initial_config"]["TicketAPI"])
        else:
            default_scenario = DEFAULT_STATE
            self.ticket_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "create_ticket":
                result = self.ticket_api.create_ticket(**parameters)
                success = True

            elif function_name == "get_ticket":
                result = self.ticket_api.get_ticket(**parameters)
                success = True

            elif function_name == "close_ticket":
                result = self.ticket_api.close_ticket(**parameters)
                success = True

            elif function_name == "resolve_ticket":
                result = self.ticket_api.resolve_ticket(**parameters)
                success = True

            elif function_name == "_find_ticket":
                result = self.ticket_api._find_ticket(**parameters)
                success = True

            elif function_name == "ticket_login":
                result = self.ticket_api.ticket_login(**parameters)
                success = True

            elif function_name == "ticket_get_login_status":
                result = self.ticket_api.ticket_get_login_status()
                success = True

            elif function_name == "logout":
                result = self.ticket_api.logout()
                success = True

            elif function_name == "get_user_tickets":
                result = self.ticket_api.get_user_tickets(**parameters)
                success = True

            else:
                result = f"Function {function_name} not found in TicketAPI"
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

        return str(result), success
