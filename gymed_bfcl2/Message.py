from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.message_api import MessageAPI, DEFAULT_STATE

class MessageEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.message_api = MessageAPI()
        self.reward_config = {
            "message_sent": 2.0,            # 成功发送消息
            "message_received": 1.5,        # 成功接收消息
            "user_authenticated": 1.0,      # 用户认证成功
            "user_found": 1.0,              # 找到用户
            "message_searched": 1.5,        # 成功搜索消息
            "task_completion": 5.0,         # 任务完成
            "send_failed": -2.0,            # 发送失败
            "user_not_found": -1.5,         # 用户未找到
            "unauthorized": -2.0,           # 未授权操作
        }
        self.current_user = None
        self.message_history = []
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        # 从 test_entry 中获取 initial_config 并加载
        if "initial_config" in self.test_entry and "MessageAPI" in self.test_entry["initial_config"]:
            # 直接传递配置到 TravelAPI
            self.message_api._load_scenario(self.test_entry["initial_config"]["MessageAPI"])
        else:
            default_scenario = DEFAULT_STATE
            self.message_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "_generate_id":
                result = self.message_api._generate_id()
                success = True

            elif function_name == "list_users":
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

        self.message_history.append(operation)

        return str(result), success