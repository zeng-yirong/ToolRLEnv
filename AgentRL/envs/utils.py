import copy
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import importlib


class FunctionCallExecutor:
    """
    函数调用执行器

    基于BFCL的multi_turn_utils.py，适配为Gymnasium环境使用的执行器。
    """

    def __init__(self, test_entry: Dict[str, Any]):
        """
        初始化函数调用执行器

        Args:
            test_entry: BFCL测试条目
        """
        self.test_entry = test_entry
        self.involved_classes = test_entry.get("involved_classes", [])
        self.initial_config = test_entry.get("initial_config", {})
        self.test_entry_id = test_entry["id"]

        # 类实例缓存
        self.instances = {}
        self.class_method_mapping = {}

        # 初始化类实例
        self._initialize_instances()

    def _initialize_instances(self):
        """初始化所有涉及的类实例"""
        for class_name in self.involved_classes:
            # 动态导入模块
            module_path = f"eval_checker.multi_turn_eval.func_source_code.travel_booking"
            try:
                module = importlib.import_module(module_path)
                class_ = getattr(module, class_name)
                instance = class_()

                # 加载初始配置
                if class_name not in ["WebSearchAPI"]:  # 无状态类
                    class_initial_config = self.initial_config.get(class_name, {})
                    if hasattr(instance, '_load_scenario'):
                        instance._load_scenario(
                            copy.deepcopy(class_initial_config),
                            long_context=False
                        )

                self.instances[class_name] = instance

                # 构建方法映射
                for method_name, method in instance.__class__.__dict__.items():
                    if callable(method) and not method_name.startswith("_"):
                        self.class_method_mapping[method_name] = class_name

            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load class {class_name}: {e}")
                continue

    def execute(self, func_calls: List[str]) -> List[str]:
        """
        执行函数调用列表

        Args:
            func_calls: 函数调用字符串列表

        Returns:
            执行结果列表
        """
        results = []

        for func_call in func_calls:
            if not func_call.strip():
                results.append("No function executed")
                continue

            try:
                # 处理函数调用字符串
                processed_call = self._process_function_call(func_call)

                # 安全执行
                result = self._safe_execute(processed_call)
                results.append(str(result))
            except Exception as e:
                results.append(f"Error during execution: {str(e)}")

        return results

    def _process_function_call(self, func_call: str) -> str:
        """处理函数调用字符串，添加实例前缀"""
        def replace_function(match):
            func_name = match.group(1)
            if func_name in self.class_method_mapping:
                class_name = self.class_method_mapping[func_name]
                instance = self.instances[class_name]
                return f"self.instances['{class_name}'].{func_name}"
            return func_name

        # 正则表达式匹配函数名
        pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"
        processed_call = re.sub(pattern, replace_function, func_call)
        return processed_call

    def _safe_execute(self, func_call: str) -> Any:
        """安全执行函数调用"""
        # 安全检查
        forbidden_functions = ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]
        func_name = func_call.split("(")[0].split(".")[-1] if "(" in func_call else func_call.split(".")[-1]

        if func_name in forbidden_functions:
            raise Exception(f"Function call {func_name} is not allowed.")

        # 执行函数调用
        try:
            # 创建局部命名空间
            local_namespace = {"self": self}
            result = eval(func_call, {"__builtins__": {}}, local_namespace)
            return result
        except Exception as e:
            raise e


class StateManager:
    """
    状态管理器

    管理环境状态，检查状态一致性等。
    """

    def __init__(self, test_entry: Dict[str, Any]):
        """
        初始化状态管理器

        Args:
            test_entry: BFCL测试条目
        """
        self.test_entry = test_entry
        self.initial_config = test_entry.get("initial_config", {})
        self.goal_state = test_entry.get("goal_state", {})
        self.current_state = {}
        self.execution_results = []

    def reset(self):
        """重置状态管理器"""
        self.current_state = {}
        self.execution_results = []

    def load_initial_config(self, config: Dict[str, Any]):
        """加载初始配置"""
        self.current_state.update(config)

    def update_from_execution(self, execution_result: str, execution_success: bool):
        """根据执行结果更新状态"""
        self.execution_results.append({
            "result": execution_result,
            "success": execution_success
        })

        # 这里可以根据执行结果更新状态
        # 具体实现需要根据不同的API类型来定

    def check_state_consistency(self) -> bool:
        """检查状态一致性"""
        # 简化实现，实际中需要更复杂的状态检查逻辑
        return True

    def is_goal_achieved(self) -> bool:
        """检查是否达到目标状态"""
        # 简化实现，实际中需要根据具体任务来定义
        if len(self.execution_results) > 0:
            last_result = self.execution_results[-1]
            return last_result["success"] and "error" not in last_result["result"].lower()
        return False

    def is_failed(self) -> bool:
        """检查是否失败"""
        if len(self.execution_results) > 0:
            last_result = self.execution_results[-1]
            return not last_result["success"]
        return False

    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "current_state": self.current_state,
            "execution_count": len(self.execution_results),
            "last_success": self.execution_results[-1]["success"] if self.execution_results else True,
            "goal_achieved": self.is_goal_achieved()
        }


class MultiTurnEvaluator:
    """
    多轮评估器

    提供与BFCL原始评估逻辑兼容的评估功能。
    """

    def __init__(self, test_entry: Dict[str, Any]):
        """
        初始化多轮评估器

        Args:
            test_entry: BFCL测试条目
        """
        self.test_entry = test_entry
        self.ground_truth = test_entry.get("ground_truth", [])
        self.executor = FunctionCallExecutor(test_entry)
        self.state_manager = StateManager(test_entry)

    def evaluate_model_response(
        self,
        model_responses: List[List[str]],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        评估模型响应

        Args:
            model_responses: 模型响应，每轮的函数调用列表
            detailed: 是否返回详细信息

        Returns:
            评估结果
        """
        total_reward = 0.0
        evaluation_details = []

        # 重置状态
        self.state_manager.reset()

        for turn_idx, (model_turn_response, ground_truth_turn) in enumerate(
            zip(model_responses, self.ground_truth)
        ):
            # 执行模型响应
            model_results = self.executor.execute(model_turn_response)

            # 执行标准答案
            ground_truth_results = self.executor.execute(ground_truth_turn)

            # 评估本轮结果
            turn_reward, turn_details = self._evaluate_turn(
                model_results, ground_truth_results, turn_idx
            )

            total_reward += turn_reward
            evaluation_details.append(turn_details)

        result = {
            "total_reward": total_reward,
            "average_reward_per_turn": total_reward / len(model_responses) if model_responses else 0,
            "success": total_reward > 0,
            "turns_evaluated": len(model_responses)
        }

        if detailed:
            result["details"] = evaluation_details

        return result

    def _evaluate_turn(
        self,
        model_results: List[str],
        ground_truth_results: List[str],
        turn_idx: int
    ) -> Tuple[float, Dict[str, Any]]:
        """评估单轮结果"""
        reward = 0.0

        # 检查执行结果是否匹配
        if self._results_match(model_results, ground_truth_results):
            reward += 1.0
        else:
            reward -= 0.5

        # 检查状态一致性
        if self.state_manager.check_state_consistency():
            reward += 0.5
        else:
            reward -= 0.3

        details = {
            "turn": turn_idx,
            "model_results": model_results,
            "ground_truth_results": ground_truth_results,
            "results_match": self._results_match(model_results, ground_truth_results),
            "state_consistent": self.state_manager.check_state_consistency(),
            "reward": reward
        }

        return reward, details

    def _results_match(self, model_results: List[str], ground_truth_results: List[str]) -> bool:
        """检查结果是否匹配"""
        # 简化实现，实际中需要更复杂的匹配逻辑
        if len(model_results) != len(ground_truth_results):
            return False

        for model_result, ground_result in zip(model_results, ground_truth_results):
            # 移除错误信息的差异比较
            model_clean = self._clean_result(model_result)
            ground_clean = self._clean_result(ground_result)

            if model_clean != ground_clean:
                return False

        return True

    def _clean_result(self, result: str) -> str:
        """清理结果字符串，移除时间戳等变量信息"""
        # 移除常见的时间戳和ID模式
        result = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z', '', result)
        result = re.sub(r'id:\s*\d+', 'id: X', result)
        result = re.sub(r'ID:\s*\d+', 'ID: X', result)
        result = re.sub(r'vec_id:\s*\d+', 'vec_id: X', result)

        return result.strip()
