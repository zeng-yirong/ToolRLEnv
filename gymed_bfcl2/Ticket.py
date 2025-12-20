from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.ticket_api import TicketAPI, DEFAULT_STATE

class TicketEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.ticket_api = TicketAPI()
        self.reward_config = {
            "ticket_created": 2.5,          # 成功创建工单
            "ticket_updated": 1.5,          # 成功更新工单
            "ticket_closed": 3.0,           # 成功关闭工单
            "ticket_found": 1.0,            # 找到工单
            "comment_added": 1.0,           # 成功添加评论
            "priority_changed": 1.5,        # 成功修改优先级
            "task_completion": 5.0,         # 任务完成
            "create_failed": -2.0,          # 创建失败
            "ticket_not_found": -1.5,       # 工单未找到
            "unauthorized": -2.0,           # 未授权操作
        }
        self.current_user = None
        self.ticket_actions = []
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        """从传入的 test_entry 加载场景到 TravelAPI"""
        # 从 test_entry 中获取 initial_config 并加载
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

        self.ticket_actions.append(operation)

        return str(result), success