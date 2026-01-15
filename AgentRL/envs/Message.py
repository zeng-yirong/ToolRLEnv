from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.message_api import MessageAPI, DEFAULT_STATE

class MessageEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.message_api = MessageAPI()
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        if "initial_config" in self.test_entry and "MessageAPI" in self.test_entry["initial_config"]:
            self.message_api._load_scenario(self.test_entry["initial_config"]["MessageAPI"])
        else:
            default_scenario = DEFAULT_STATE
            self.message_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "list_users":
                result = self.message_api.list_users()
                success = True

            elif function_name == "get_user_id":
                result = self.message_api.get_user_id(**parameters)
                success = True

            elif function_name == "message_login":
                result = self.message_api.message_login(**parameters)
                success = True

            elif function_name == "message_get_login_status":
                result = self.message_api.message_get_login_status()
                success = True

            elif function_name == "send_message":
                result = self.message_api.send_message(**parameters)
                success = True

            elif function_name == "delete_message":
                result = self.message_api.delete_message(**parameters)
                success = True

            elif function_name == "view_messages_sent":
                result = self.message_api.view_messages_sent()
                success = True

            elif function_name == "add_contact":
                result = self.message_api.add_contact(**parameters)
                success = True

            elif function_name == "search_messages":
                result = self.message_api.search_messages(**parameters)
                success = True

            elif function_name == "get_message_stats":
                result = self.message_api.get_message_stats()
                success = True

            else:
                result = f"Function {function_name} not found in MessageAPI"
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
