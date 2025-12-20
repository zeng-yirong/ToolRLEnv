"""
Gorilla文件系统环境的Gymnasium适配

基于GorillaFileSystem API实现的文件系统操作环境，
支持基本的文件和目录操作，如导航、创建、读写、搜索等。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class GorillaFileSystemEnv(FunctionCallingEnv):
    """
    Gorilla文件系统操作环境

    支持以下文件系统操作：
    - 目录导航：cd, ls, pwd
    - 文件操作：touch, cat, echo, rm
    - 文件搜索：find, grep
    - 文件管理：cp, mv, sort
    - 内容查看：head, tail
    - 目录操作：mkdir, rmdir
    - 文件比较：diff
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "file_operations",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化Gorilla文件系统环境

        Args:
            test_entry: 测试条目，包含文件系统初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入GorillaFileSystem API
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import GorillaFileSystem

        self.api = GorillaFileSystem()

        # 默认奖励配置
        default_reward_config = {
            "operation_success": 2.0,      # 操作成功
            "file_found": 1.5,             # 找到目标文件
            "directory_navigated": 1.0,    # 成功导航目录
            "content_processed": 1.5,      # 成功处理文件内容
            "task_completion": 5.0,        # 任务完成
            "operation_failed": -1.0,      # 操作失败
            "file_not_found": -1.5,        # 文件未找到
            "invalid_path": -2.0,          # 无效路径
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)

        # 加载文件系统初始状态
        if test_entry and "initial_config" in test_entry:
            if "GorillaFileSystem" in test_entry["initial_config"]:
                self.api._load_scenario(test_entry["initial_config"]["GorillaFileSystem"])

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        # 基于GorillaFileSystem API的实际函数
        actions = [
            "cd", "ls", "pwd", "touch", "cat", "echo", "rm",
            "find", "grep", "cp", "mv", "sort", "head", "tail",
            "mkdir", "rmdir", "diff"
        ]
        return spaces.Discrete(len(actions))

    def _get_observation_space(self) -> spaces.Space:
        """定义观察空间"""
        return spaces.Dict({
            "question": spaces.Text(2000),
            "current_directory": spaces.Text(500),
            "directory_contents": spaces.Sequence(spaces.Text(200)),
            "available_files": spaces.Sequence(spaces.Text(200)),
            "file_system_info": spaces.Dict({
                "current_path": spaces.Text(500),
                "file_count": spaces.Discrete(100),
                "directory_count": spaces.Discrete(50)
            }),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "task_progress": spaces.Dict({
                "operations_completed": spaces.Discrete(20),
                "files_processed": spaces.Discrete(50),
                "directories_visited": spaces.Discrete(20)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "cd",
                "description": "Change the current directory",
                "parameters": {
                    "path": "string - directory path to navigate to"
                }
            },
            {
                "name": "ls",
                "description": "List contents of current directory",
                "parameters": {
                    "path": "optional string - directory path to list, defaults to current"
                }
            },
            {
                "name": "pwd",
                "description": "Print working directory",
                "parameters": {}
            },
            {
                "name": "touch",
                "description": "Create a new file",
                "parameters": {
                    "file_name": "string - name of the file to create"
                }
            },
            {
                "name": "cat",
                "description": "Read and display file contents",
                "parameters": {
                    "file_name": "string - name of the file to read"
                }
            },
            {
                "name": "echo",
                "description": "Write text to a file",
                "parameters": {
                    "text": "string - text to write",
                    "file_name": "string - file to write to"
                }
            },
            {
                "name": "rm",
                "description": "Remove a file",
                "parameters": {
                    "file_name": "string - file to remove"
                }
            },
            {
                "name": "find",
                "description": "Find files by name pattern",
                "parameters": {
                    "name_pattern": "string - pattern to search for",
                    "path": "optional string - directory to search in"
                }
            },
            {
                "name": "grep",
                "description": "Search for text pattern in files",
                "parameters": {
                    "pattern": "string - text pattern to search for",
                    "file_name": "string - file to search in"
                }
            },
            {
                "name": "cp",
                "description": "Copy files or directories",
                "parameters": {
                    "source": "string - source file/directory",
                    "destination": "string - destination path"
                }
            },
            {
                "name": "mv",
                "description": "Move or rename files",
                "parameters": {
                    "source": "string - source file/directory",
                    "destination": "string - destination path"
                }
            },
            {
                "name": "sort",
                "description": "Sort lines in a file",
                "parameters": {
                    "file_name": "string - file to sort"
                }
            },
            {
                "name": "head",
                "description": "Display first lines of a file",
                "parameters": {
                    "file_name": "string - file to read",
                    "lines": "optional int - number of lines to show"
                }
            },
            {
                "name": "tail",
                "description": "Display last lines of a file",
                "parameters": {
                    "file_name": "string - file to read",
                    "lines": "optional int - number of lines to show"
                }
            },
            {
                "name": "mkdir",
                "description": "Create a new directory",
                "parameters": {
                    "dir_name": "string - name of directory to create"
                }
            },
            {
                "name": "rmdir",
                "description": "Remove a directory",
                "parameters": {
                    "dir_name": "string - directory to remove"
                }
            },
            {
                "name": "diff",
                "description": "Compare two files line by line",
                "parameters": {
                    "file1": "string - first file",
                    "file2": "string - second file"
                }
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "cd", "ls", "pwd", "touch", "cat", "echo", "rm",
            "find", "grep", "cp", "mv", "sort", "head", "tail",
            "mkdir", "rmdir", "diff"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            if action_name == "cd":
                # 简化实现：导航到指定目录
                path = self._extract_parameter_from_question("path") or "."
                result = self._simulate_cd(path)

            elif action_name == "ls":
                path = self._extract_parameter_from_question("path") or "."
                result = self._simulate_ls(path)

            elif action_name == "pwd":
                result = self._simulate_pwd()

            elif action_name == "touch":
                file_name = self._extract_parameter_from_question("file_name")
                if file_name:
                    result = self._simulate_touch(file_name)

            elif action_name == "cat":
                file_name = self._extract_parameter_from_question("file_name")
                if file_name:
                    result = self._simulate_cat(file_name)

            elif action_name == "echo":
                text = self._extract_parameter_from_question("text")
                file_name = self._extract_parameter_from_question("file_name")
                if text and file_name:
                    result = self._simulate_echo(text, file_name)

            elif action_name == "rm":
                file_name = self._extract_parameter_from_question("file_name")
                if file_name:
                    result = self._simulate_rm(file_name)

            elif action_name == "find":
                name_pattern = self._extract_parameter_from_question("name_pattern")
                path = self._extract_parameter_from_question("path") or "."
                if name_pattern:
                    result = self._simulate_find(name_pattern, path)

            elif action_name == "grep":
                pattern = self._extract_parameter_from_question("pattern")
                file_name = self._extract_parameter_from_question("file_name")
                if pattern and file_name:
                    result = self._simulate_grep(pattern, file_name)

            elif action_name == "cp":
                source = self._extract_parameter_from_question("source")
                destination = self._extract_parameter_from_question("destination")
                if source and destination:
                    result = self._simulate_cp(source, destination)

            elif action_name == "mv":
                source = self._extract_parameter_from_question("source")
                destination = self._extract_parameter_from_question("destination")
                if source and destination:
                    result = self._simulate_mv(source, destination)

            elif action_name == "sort":
                file_name = self._extract_parameter_from_question("file_name")
                if file_name:
                    result = self._simulate_sort(file_name)

            elif action_name == "head":
                file_name = self._extract_parameter_from_question("file_name")
                lines = int(self._extract_parameter_from_question("lines") or "10")
                if file_name:
                    result = self._simulate_head(file_name, lines)

            elif action_name == "tail":
                file_name = self._extract_parameter_from_question("file_name")
                lines = int(self._extract_parameter_from_question("lines") or "10")
                if file_name:
                    result = self._simulate_tail(file_name, lines)

            elif action_name == "mkdir":
                dir_name = self._extract_parameter_from_question("dir_name")
                if dir_name:
                    result = self._simulate_mkdir(dir_name)

            elif action_name == "rmdir":
                dir_name = self._extract_parameter_from_question("dir_name")
                if dir_name:
                    result = self._simulate_rmdir(dir_name)

            elif action_name == "diff":
                file1 = self._extract_parameter_from_question("file1")
                file2 = self._extract_parameter_from_question("file2")
                if file1 and file2:
                    result = self._simulate_diff(file1, file2)

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["operation_failed"]

    # 文件系统操作模拟方法
    def _simulate_cd(self, path: str) -> Dict:
        """模拟cd操作"""
        return {"success": True, "message": f"Changed directory to {path}", "data": {"path": path}}

    def _simulate_ls(self, path: str) -> Dict:
        """模拟ls操作"""
        # 简化实现：返回一些示例文件和目录
        contents = ["file1.txt", "file2.txt", "directory1", "directory2"]
        return {"success": True, "message": f"Contents of {path}:", "data": {"contents": contents}}

    def _simulate_pwd(self) -> Dict:
        """模拟pwd操作"""
        return {"success": True, "message": "/current/directory", "data": {"path": "/current/directory"}}

    def _simulate_touch(self, file_name: str) -> Dict:
        """模拟touch操作"""
        return {"success": True, "message": f"Created file: {file_name}", "data": {"file": file_name}}

    def _simulate_cat(self, file_name: str) -> Dict:
        """模拟cat操作"""
        # 简化实现：返回示例内容
        content = f"Sample content of {file_name}"
        return {"success": True, "message": content, "data": {"content": content, "file": file_name}}

    def _simulate_echo(self, text: str, file_name: str) -> Dict:
        """模拟echo操作"""
        return {"success": True, "message": f"Written '{text}' to {file_name}", "data": {"file": file_name, "text": text}}

    def _simulate_rm(self, file_name: str) -> Dict:
        """模拟rm操作"""
        return {"success": True, "message": f"Removed file: {file_name}", "data": {"file": file_name}}

    def _simulate_find(self, name_pattern: str, path: str) -> Dict:
        """模拟find操作"""
        # 简化实现：返回一些匹配的文件
        found_files = [f"{path}/{name_pattern}_file1.txt", f"{path}/{name_pattern}_file2.txt"]
        return {"success": True, "message": f"Found files matching '{name_pattern}':", "data": {"files": found_files}}

    def _simulate_grep(self, pattern: str, file_name: str) -> Dict:
        """模拟grep操作"""
        # 简化实现：返回匹配的行
        matching_lines = [f"Line 1: contains {pattern}", f"Line 3: contains {pattern}"]
        return {"success": True, "message": f"Lines matching '{pattern}' in {file_name}:", "data": {"lines": matching_lines, "file": file_name}}

    def _simulate_cp(self, source: str, destination: str) -> Dict:
        """模拟cp操作"""
        return {"success": True, "message": f"Copied {source} to {destination}", "data": {"source": source, "destination": destination}}

    def _simulate_mv(self, source: str, destination: str) -> Dict:
        """模拟mv操作"""
        return {"success": True, "message": f"Moved {source} to {destination}", "data": {"source": source, "destination": destination}}

    def _simulate_sort(self, file_name: str) -> Dict:
        """模拟sort操作"""
        return {"success": True, "message": f"Sorted {file_name}", "data": {"file": file_name}}

    def _simulate_head(self, file_name: str, lines: int) -> Dict:
        """模拟head操作"""
        # 简化实现：返回前几行
        content_lines = [f"Line {i+1} of {file_name}" for i in range(min(lines, 5))]
        return {"success": True, "message": f"First {lines} lines of {file_name}:", "data": {"lines": content_lines, "file": file_name}}

    def _simulate_tail(self, file_name: str, lines: int) -> Dict:
        """模拟tail操作"""
        # 简化实现：返回后几行
        content_lines = [f"Line {i+1} of {file_name}" for i in range(max(1, 5-lines), 5)]
        return {"success": True, "message": f"Last {lines} lines of {file_name}:", "data": {"lines": content_lines, "file": file_name}}

    def _simulate_mkdir(self, dir_name: str) -> Dict:
        """模拟mkdir操作"""
        return {"success": True, "message": f"Created directory: {dir_name}", "data": {"directory": dir_name}}

    def _simulate_rmdir(self, dir_name: str) -> Dict:
        """模拟rmdir操作"""
        return {"success": True, "message": f"Removed directory: {dir_name}", "data": {"directory": dir_name}}

    def _simulate_diff(self, file1: str, file2: str) -> Dict:
        """模拟diff操作"""
        # 简化实现：返回一些差异
        differences = [f"Line 2: {file1} has different content than {file2}"]
        return {"success": True, "message": f"Differences between {file1} and {file2}:", "data": {"differences": differences, "files": [file1, file2]}}

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            return self.reward_config["operation_failed"]

        # 基础操作成功奖励
        reward = self.reward_config["operation_success"]

        # 特定操作的额外奖励
        if action_name in ["find", "grep"]:
            reward += self.reward_config["file_found"]
        elif action_name == "cd":
            reward += self.reward_config["directory_navigated"]
        elif action_name in ["cat", "echo", "sort", "head", "tail"]:
            reward += self.reward_config["content_processed"]

        return reward

    def _extract_parameter_from_question(self, param_name: str) -> Optional[str]:
        """从问题中提取参数（简化实现）"""
        # 这是一个简化的参数提取方法
        # 在实际应用中，这里应该有更复杂的NLP逻辑来解析用户意图
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 简单的字符串匹配（实际应用中需要更智能的解析）
        if param_name == "file_name" and ".txt" in question:
            # 提取文件名
            words = question.split()
            for word in words:
                if word.endswith(".txt") or word.endswith(".py") or word.endswith(".md"):
                    return word
        elif param_name == "dir_name" and "directory" in question.lower():
            # 提取目录名
            if "temp" in question.lower():
                return "temp"
            elif "archive" in question.lower():
                return "archive"
        elif param_name == "text":
            # 提取要写入的文本（在引号中的内容）
            import re
            matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", question)
            if matches:
                return matches[0][0] if matches[0][0] else matches[0][1]

        return None

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        return {
            "question": self.current_question,
            "current_directory": "/current/directory",  # 简化实现
            "directory_contents": ["file1.txt", "file2.txt", "directory1"],  # 简化实现
            "available_files": ["file1.txt", "file2.txt"],  # 简化实现
            "file_system_info": {
                "current_path": "/current/directory",
                "file_count": 2,
                "directory_count": 1
            },
            "function_docs": self._get_function_docs(),
            "task_progress": {
                "operations_completed": getattr(self, 'operations_completed', 0),
                "files_processed": getattr(self, 'files_processed', 0),
                "directories_visited": getattr(self, 'directories_visited', 0)
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # 简化的任务完成检查
        # 在实际应用中，这里应该检查是否完成了用户要求的所有文件操作
        progress = obs["task_progress"]
        return progress["operations_completed"] >= 3  # 假设完成3个操作即为任务完成