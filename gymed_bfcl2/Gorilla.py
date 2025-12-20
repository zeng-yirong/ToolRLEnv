from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import GorillaFileSystem,DEFAULT_STATE

class GorillaEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.gorilla_api = GorillaFileSystem()
        self.reward_config = {
            "operation_success": 2.0,      # 操作成功
            "file_found": 1.5,             # 找到目标文件
            "directory_navigated": 1.0,    # 成功导航目录
            "content_processed": 1.5,      # 成功处理文件内容
            "task_completion": 5.0,        # 任务完成
            "operation_failed": -1.0,      # 操作失败
            "file_not_found": -1.5,        # 文件未找到
            "invalid_path": -2.0,          # 无效路径
        }
        self.gorilla_actions = []
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        """从传入的 test_entry 加载场景到 TravelAPI"""
        # 从 test_entry 中获取 initial_config 并加载
        if "initial_config" in self.test_entry and "GorillaFileSystem" in self.test_entry["initial_config"]:
            # 直接传递配置到 TravelAPI
            self.gorilla_api._load_scenario(self.test_entry["initial_config"]["GorillaFileSystem"])
        else:
            default_scenario = DEFAULT_STATE
            self.gorilla_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "_load_directory":
                result = self.gorilla_api._load_directory(**parameters)
                success = True

            elif function_name == "_populate_directory":
                result = self.gorilla_api._populate_directory(**parameters)
                success = True

            elif function_name == "pwd":
                result = self.gorilla_api.pwd()
                success = True

            elif function_name == "ls":
                result = self.gorilla_api.ls()
                success = True

            elif function_name == "cd":
                result = self.gorilla_api.cd(**parameters)
                success = True

            elif function_name == "_validate_file_or_directory_name":
                result = self.gorilla_api._validate_file_or_directory_name(**parameters)
                success = True

            elif function_name == "mkdir":
                result = self.gorilla_api.mkdir(**parameters)
                success = True

            elif function_name == "touch":
                result = self.gorilla_api.touch(**parameters)
                success = True

            elif function_name == "echo":
                result = self.gorilla_api.echo(**parameters)
                success = True

            elif function_name == "cat":
                result = self.gorilla_api.cat(**parameters)
                success = True

            elif function_name == "find":
                result = self.gorilla_api.find(**parameters)
                success = True

            elif function_name == "wc":
                result = self.gorilla_api.wc(**parameters)
                success = True

            elif function_name == "sort":
                result = self.gorilla_api.sort(**parameters)
                success = True

            elif function_name == "grep":
                result = self.gorilla_api.grep(**parameters)
                success = True

            elif function_name == "du":
                result = self.gorilla_api.du(**parameters)
                success = True

            elif function_name == "tail":
                result = self.gorilla_api.tail(**parameters)
                success = True

            elif function_name == "diff":
                result = self.gorilla_api.diff(**parameters)
                success = True

            elif function_name == "mv":
                result = self.gorilla_api.mv(**parameters)
                success = True

            elif function_name == "rm":
                result = self.gorilla_api.rm(**parameters)
                success = True

            elif function_name == "rmdir":
                result = self.gorilla_api.rmdir(**parameters)
                success = True

            elif function_name == "cp":
                result = self.gorilla_api.cp(**parameters)
                success = True

            elif function_name == "_navigate_to_directory":
                result = self.gorilla_api._navigate_to_directory(**parameters)
                success = True

            elif function_name == "_parse_positions":
                result = self.gorilla_api._parse_positions(**parameters)
                success = True

            else:
                result = f"Function {function_name} not found in GorillaFileSystem"
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

        self.gorilla_actions.append(operation)

        return str(result), success