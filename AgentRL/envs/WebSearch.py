from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI

class WebSearchEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.web_api = WebSearchAPI()
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        pass

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "search_engine_query":
                result = self.web_api.search_engine_query(**parameters)
                success = True

            elif function_name == "fetch_url_content":
                result = self.web_api.fetch_url_content(**parameters)
                success = True

            elif function_name == "_fake_requests_get_error_msg":
                result = self.web_api._fake_requests_get_error_msg(**parameters)
                success = True

            else:
                result = f"Function {function_name} not found in WebSearchAPI"
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

