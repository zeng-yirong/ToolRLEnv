from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI

class WebSearchEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.web_api = WebSearchAPI()
        self.reward_config = {
            "search_completed": 2.0,  # 搜索完成
            "relevant_found": 3.0,  # 找到相关结果
            "page_browsed": 1.5,  # 网页浏览成功
            "information_extracted": 2.5,  # 信息提取成功
            "results_filtered": 1.0,  # 结果过滤成功
            "task_completion": 5.0,  # 任务完成
            "search_failed": -2.0,  # 搜索失败
            "no_results": -1.5,  # 无搜索结果
            "access_denied": -2.5,  # 访问被拒绝
        }
        self.search_history = []
        self.browsed_pages = []
        self.saved_results = []
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        """从传入的 test_entry 加载场景到 TravelAPI"""
        if "initial_config" in self.test_entry and "WebSearchAPI" in self.test_entry["initial_config"]:
            self.web_api._load_scenario(self.test_entry["initial_config"]["WebSearchAPI"])
        else:
            default_scenario = {
                "show_snippet":True
            }
            self.web_api._load_scenario(default_scenario)

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

        self.search_history.append(operation)

        return str(result), success
