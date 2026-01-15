from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.file_system import FileSystem,DEFAULT_STATE

class FileEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.file_api = FileSystem()
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        if "initial_config" in self.test_entry and "File" in self.test_entry["initial_config"]:
            self.file_api._load_scenario(self.test_entry["initial_config"]["File"])
        else:
            default_scenario = DEFAULT_STATE
            self.file_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:

            if function_name == "_populate_directory":
                result = self.file_api._populate_directory(**parameters)
                success = True

            elif function_name == "pwd":
                result = self.file_api.pwd()
                success = True

            elif function_name == "ls":
                result = self.file_api.ls()
                success = True

            elif function_name == "cd":
                result = self.file_api.cd(**parameters)
                success = True

            elif function_name == "_validate_file_or_directory_name":
                result = self.file_api._validate_file_or_directory_name(**parameters)
                success = True

            elif function_name == "mkdir":
                result = self.file_api.mkdir(**parameters)
                success = True

            elif function_name == "touch":
                result = self.file_api.touch(**parameters)
                success = True

            elif function_name == "echo":
                result = self.file_api.echo(**parameters)
                success = True

            elif function_name == "cat":
                result = self.file_api.cat(**parameters)
                success = True

            elif function_name == "find":
                result = self.file_api.find(**parameters)
                success = True

            elif function_name == "wc":
                result = self.file_api.wc(**parameters)
                success = True

            elif function_name == "sort":
                result = self.file_api.sort(**parameters)
                success = True

            elif function_name == "grep":
                result = self.file_api.grep(**parameters)
                success = True

            elif function_name == "du":
                result = self.file_api.du(**parameters)
                success = True

            elif function_name == "tail":
                result = self.file_api.tail(**parameters)
                success = True

            elif function_name == "diff":
                result = self.file_api.diff(**parameters)
                success = True

            elif function_name == "mv":
                result = self.file_api.mv(**parameters)
                success = True

            elif function_name == "rm":
                result = self.file_api.rm(**parameters)
                success = True

            elif function_name == "rmdir":
                result = self.file_api.rmdir(**parameters)
                success = True

            elif function_name == "cp":
                result = self.file_api.cp(**parameters)
                success = True

            elif function_name == "_navigate_to_directory":
                result = self.file_api._navigate_to_directory(**parameters)
                success = True

            elif function_name == "_parse_positions":
                result = self.file_api._parse_positions(**parameters)
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
