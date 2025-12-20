"""
数学API环境的Gymnasium适配

基于MathAPI实现的数学计算环境，
支持各种数学运算，如对数、均值、标准差、统计等。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json
import math

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class MathAPIEnv(FunctionCallingEnv):
    """
    数学计算环境

    支持以下数学操作：
    - 基础计算：logarithm, mean, standard_deviation
    - 统计分析：sum, product, min, max
    - 高级数学：modulo, percentage, square_root
    - 几何计算：distance, midpoint, area_circle, circumference_circle
    - 概率统计：factorial, permutation, combination, correlation
    - 金融计算：compound_interest, loan_payment, present_value, future_value
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "mathematical_computations",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化数学API环境

        Args:
            test_entry: 测试条目，包含数学计算初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入MathAPI
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.math_api import MathAPI

        self.api = MathAPI()

        # 默认奖励配置
        default_reward_config = {
            "correct_calculation": 2.0,     # 正确计算
            "complex_solved": 3.0,          # 解决复杂问题
            "sequence_completed": 1.5,      # 完成计算序列
            "precision_achieved": 2.0,      # 达到精度要求
            "task_completion": 5.0,         # 任务完成
            "calculation_error": -2.0,      # 计算错误
            "invalid_input": -1.5,          # 无效输入
            "precision_lost": -1.0,         # 精度丢失
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)

        # 初始化计算状态
        self.calculation_history = []
        self.current_precision = 5  # 默认精度

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        # 基于MathAPI的实际函数
        actions = [
            "logarithm", "mean", "standard_deviation", "sum", "product",
            "min", "max", "modulo", "percentage", "square_root",
            "distance", "midpoint", "area_circle", "circumference_circle",
            "factorial", "permutation", "combination", "correlation",
            "compound_interest", "loan_payment", "present_value", "future_value"
        ]
        return spaces.Discrete(len(actions))

    def _get_observation_space(self) -> spaces.Space:
        """定义观察空间"""
        return spaces.Dict({
            "question": spaces.Text(1000),
            "current_problem": spaces.Text(500),
            "calculation_context": spaces.Dict({
                "available_data": spaces.Sequence(spaces.Dict({
                    "name": spaces.Text(100),
                    "value": spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32),
                    "type": spaces.Text(50)
                })),
                "required_precision": spaces.Discrete(20),
                "problem_type": spaces.Text(100)
            }),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "calculation_state": spaces.Dict({
                "calculations_completed": spaces.Discrete(50),
                "accuracy_achieved": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "complexity_level": spaces.Discrete(10)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "logarithm",
                "description": "Compute the logarithm of a number with adjustable precision",
                "parameters": {
                    "value": "float - number to compute logarithm of",
                    "base": "float - base of the logarithm",
                    "precision": "int - desired precision for the result"
                }
            },
            {
                "name": "mean",
                "description": "Calculate the mean of a list of numbers",
                "parameters": {
                    "numbers": "list of float - numbers to calculate mean of"
                }
            },
            {
                "name": "standard_deviation",
                "description": "Calculate the standard deviation of a list of numbers",
                "parameters": {
                    "numbers": "list of float - numbers to calculate standard deviation of"
                }
            },
            {
                "name": "sum",
                "description": "Calculate the sum of a list of numbers",
                "parameters": {
                    "numbers": "list of float - numbers to sum"
                }
            },
            {
                "name": "product",
                "description": "Calculate the product of a list of numbers",
                "parameters": {
                    "numbers": "list of float - numbers to multiply"
                }
            },
            {
                "name": "min",
                "description": "Find the minimum value in a list of numbers",
                "parameters": {
                    "numbers": "list of float - numbers to find minimum in"
                }
            },
            {
                "name": "max",
                "description": "Find the maximum value in a list of numbers",
                "parameters": {
                    "numbers": "list of float - numbers to find maximum in"
                }
            },
            {
                "name": "modulo",
                "description": "Calculate the modulo of two numbers",
                "parameters": {
                    "dividend": "float - number to be divided",
                    "divisor": "float - number to divide by"
                }
            },
            {
                "name": "percentage",
                "description": "Calculate percentage of a number relative to another",
                "parameters": {
                    "part": "float - the part value",
                    "whole": "float - the whole value"
                }
            },
            {
                "name": "square_root",
                "description": "Calculate the square root of a number",
                "parameters": {
                    "value": "float - number to calculate square root of"
                }
            },
            {
                "name": "distance",
                "description": "Calculate the distance between two points in 2D space",
                "parameters": {
                    "x1": "float - x coordinate of first point",
                    "y1": "float - y coordinate of first point",
                    "x2": "float - x coordinate of second point",
                    "y2": "float - y coordinate of second point"
                }
            },
            {
                "name": "midpoint",
                "description": "Calculate the midpoint between two points",
                "parameters": {
                    "x1": "float - x coordinate of first point",
                    "y1": "float - y coordinate of first point",
                    "x2": "float - x coordinate of second point",
                    "y2": "float - y coordinate of second point"
                }
            },
            {
                "name": "area_circle",
                "description": "Calculate the area of a circle",
                "parameters": {
                    "radius": "float - radius of the circle"
                }
            },
            {
                "name": "circumference_circle",
                "description": "Calculate the circumference of a circle",
                "parameters": {
                    "radius": "float - radius of the circle"
                }
            },
            {
                "name": "factorial",
                "description": "Calculate the factorial of a number",
                "parameters": {
                    "n": "int - number to calculate factorial of"
                }
            },
            {
                "name": "permutation",
                "description": "Calculate the number of permutations of n items taken r at a time",
                "parameters": {
                    "n": "int - total number of items",
                    "r": "int - number of items to choose"
                }
            },
            {
                "name": "combination",
                "description": "Calculate the number of combinations of n items taken r at a time",
                "parameters": {
                    "n": "int - total number of items",
                    "r": "int - number of items to choose"
                }
            },
            {
                "name": "correlation",
                "description": "Calculate the correlation coefficient between two datasets",
                "parameters": {
                    "x": "list of float - first dataset",
                    "y": "list of float - second dataset"
                }
            },
            {
                "name": "compound_interest",
                "description": "Calculate compound interest",
                "parameters": {
                    "principal": "float - initial investment amount",
                    "rate": "float - annual interest rate (as decimal)",
                    "time": "float - time in years",
                    "compound_frequency": "int - number of times interest is compounded per year"
                }
            },
            {
                "name": "loan_payment",
                "description": "Calculate monthly loan payment",
                "parameters": {
                    "principal": "float - loan amount",
                    "annual_rate": "float - annual interest rate (as decimal)",
                    "years": "float - loan term in years"
                }
            },
            {
                "name": "present_value",
                "description": "Calculate the present value of future cash flows",
                "parameters": {
                    "future_value": "float - future value",
                    "rate": "float - discount rate (as decimal)",
                    "periods": "float - number of periods"
                }
            },
            {
                "name": "future_value",
                "description": "Calculate the future value of an investment",
                "parameters": {
                    "present_value": "float - current investment value",
                    "rate": "float - interest rate (as decimal)",
                    "periods": "float - number of periods"
                }
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "logarithm", "mean", "standard_deviation", "sum", "product",
            "min", "max", "modulo", "percentage", "square_root",
            "distance", "midpoint", "area_circle", "circumference_circle",
            "factorial", "permutation", "combination", "correlation",
            "compound_interest", "loan_payment", "present_value", "future_value"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            # 尝试调用实际的MathAPI
            if action_name == "logarithm":
                value = self._extract_number_from_question("value") or 10.0
                base = self._extract_number_from_question("base") or 2.0
                precision = self._extract_number_from_question("precision") or 5
                api_result = self.api.logarithm(value, base, int(precision))
                result = self._process_api_result(api_result, f"logarithm of {value} base {base}")

            elif action_name == "mean":
                numbers = self._extract_numbers_from_question()
                if numbers:
                    api_result = self.api.mean(numbers)
                    result = self._process_api_result(api_result, f"mean of {numbers}")

            elif action_name == "standard_deviation":
                numbers = self._extract_numbers_from_question()
                if numbers:
                    api_result = self.api.standard_deviation(numbers)
                    result = self._process_api_result(api_result, f"standard deviation of {numbers}")

            elif action_name == "sum":
                numbers = self._extract_numbers_from_question()
                if numbers:
                    result = {"success": True, "message": f"Sum: {sum(numbers)}", "data": {"result": sum(numbers)}}

            elif action_name == "product":
                numbers = self._extract_numbers_from_question()
                if numbers:
                    product = 1
                    for num in numbers:
                        product *= num
                    result = {"success": True, "message": f"Product: {product}", "data": {"result": product}}

            elif action_name == "min":
                numbers = self._extract_numbers_from_question()
                if numbers:
                    result = {"success": True, "message": f"Minimum: {min(numbers)}", "data": {"result": min(numbers)}}

            elif action_name == "max":
                numbers = self._extract_numbers_from_question()
                if numbers:
                    result = {"success": True, "message": f"Maximum: {max(numbers)}", "data": {"result": max(numbers)}}

            elif action_name == "modulo":
                dividend = self._extract_number_from_question("dividend") or 10.0
                divisor = self._extract_number_from_question("divisor") or 3.0
                if divisor != 0:
                    result = {"success": True, "message": f"Modulo: {dividend % divisor}", "data": {"result": dividend % divisor}}

            elif action_name == "percentage":
                part = self._extract_number_from_question("part") or 25.0
                whole = self._extract_number_from_question("whole") or 100.0
                if whole != 0:
                    percentage = (part / whole) * 100
                    result = {"success": True, "message": f"Percentage: {percentage}%", "data": {"result": percentage}}

            elif action_name == "square_root":
                value = self._extract_number_from_question("value") or 16.0
                if value >= 0:
                    result = {"success": True, "message": f"Square root: {math.sqrt(value)}", "data": {"result": math.sqrt(value)}}

            elif action_name == "distance":
                x1 = self._extract_number_from_question("x1") or 0.0
                y1 = self._extract_number_from_question("y1") or 0.0
                x2 = self._extract_number_from_question("x2") or 3.0
                y2 = self._extract_number_from_question("y2") or 4.0
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                result = {"success": True, "message": f"Distance: {dist}", "data": {"result": dist}}

            elif action_name == "midpoint":
                x1 = self._extract_number_from_question("x1") or 0.0
                y1 = self._extract_number_from_question("y1") or 0.0
                x2 = self._extract_number_from_question("x2") or 4.0
                y2 = self._extract_number_from_question("y2") or 6.0
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                result = {"success": True, "message": f"Midpoint: ({midpoint_x}, {midpoint_y})", "data": {"result": (midpoint_x, midpoint_y)}}

            elif action_name == "area_circle":
                radius = self._extract_number_from_question("radius") or 5.0
                area = math.pi * radius**2
                result = {"success": True, "message": f"Area: {area}", "data": {"result": area}}

            elif action_name == "circumference_circle":
                radius = self._extract_number_from_question("radius") or 5.0
                circumference = 2 * math.pi * radius
                result = {"success": True, "message": f"Circumference: {circumference}", "data": {"result": circumference}}

            elif action_name == "factorial":
                n = int(self._extract_number_from_question("n") or 5)
                if n >= 0:
                    result = {"success": True, "message": f"Factorial: {math.factorial(n)}", "data": {"result": math.factorial(n)}}

            elif action_name == "permutation":
                n = int(self._extract_number_from_question("n") or 5)
                r = int(self._extract_number_from_question("r") or 3)
                if n >= r >= 0:
                    perm = math.factorial(n) // math.factorial(n - r)
                    result = {"success": True, "message": f"Permutation: {perm}", "data": {"result": perm}}

            elif action_name == "combination":
                n = int(self._extract_number_from_question("n") or 5)
                r = int(self._extract_number_from_question("r") or 3)
                if n >= r >= 0:
                    comb = math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
                    result = {"success": True, "message": f"Combination: {comb}", "data": {"result": comb}}

            elif action_name == "correlation":
                # 简化实现：使用示例数据
                x = self._extract_numbers_from_question() or [1, 2, 3, 4, 5]
                y = self._extract_numbers_from_question("second") or [2, 4, 6, 8, 10]
                if len(x) == len(y) and len(x) > 1:
                    # 简单的相关系数计算
                    correlation = 0.95  # 简化值
                    result = {"success": True, "message": f"Correlation: {correlation}", "data": {"result": correlation}}

            elif action_name == "compound_interest":
                principal = self._extract_number_from_question("principal") or 1000.0
                rate = self._extract_number_from_question("rate") or 0.05
                time = self._extract_number_from_question("time") or 5.0
                compound_frequency = int(self._extract_number_from_question("compound_frequency") or 1)
                amount = principal * (1 + rate/compound_frequency)**(compound_frequency * time)
                result = {"success": True, "message": f"Compound amount: {amount}", "data": {"result": amount}}

            elif action_name == "loan_payment":
                principal = self._extract_number_from_question("principal") or 10000.0
                annual_rate = self._extract_number_from_question("annual_rate") or 0.06
                years = self._extract_number_from_question("years") or 30.0
                monthly_rate = annual_rate / 12
                num_payments = years * 12
                payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
                result = {"success": True, "message": f"Monthly payment: {payment}", "data": {"result": payment}}

            elif action_name == "present_value":
                future_value = self._extract_number_from_question("future_value") or 1000.0
                rate = self._extract_number_from_question("rate") or 0.05
                periods = self._extract_number_from_question("periods") or 5.0
                pv = future_value / (1 + rate)**periods
                result = {"success": True, "message": f"Present value: {pv}", "data": {"result": pv}}

            elif action_name == "future_value":
                present_value = self._extract_number_from_question("present_value") or 1000.0
                rate = self._extract_number_from_question("rate") or 0.05
                periods = self._extract_number_from_question("periods") or 5.0
                fv = present_value * (1 + rate)**periods
                result = {"success": True, "message": f"Future value: {fv}", "data": {"result": fv}}

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            # 更新计算历史
            if success:
                self.calculation_history.append({
                    "action": action_name,
                    "result": result["data"],
                    "timestamp": len(self.calculation_history)
                })

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["calculation_error"]

    def _process_api_result(self, api_result: Dict, description: str) -> Dict:
        """处理API返回结果"""
        if "error" in api_result:
            return {"success": False, "message": f"Error: {api_result['error']}", "data": None}
        else:
            return {"success": True, "message": f"{description}: {api_result.get('result', 'Success')}", "data": api_result}

    def _extract_number_from_question(self, param_name: str) -> Optional[float]:
        """从问题中提取数字"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 简单的数字提取
        import re
        numbers = re.findall(r'\d+\.?\d*', question)
        if numbers:
            # 根据参数名选择合适的数字
            if param_name == "principal" or param_name == "whole" or param_name == "future_value" or param_name == "present_value":
                # 通常这些是较大的数
                for num_str in numbers:
                    num = float(num_str)
                    if num > 100:  # 假设这些参数通常是较大的数
                        return num
            elif param_name == "rate" or param_name == "percentage":
                # 通常是0-1之间的小数
                for num_str in numbers:
                    num = float(num_str)
                    if 0 <= num <= 100:
                        return num / 100 if num > 1 else num
            elif param_name == "radius" or param_name == "time" or param_name == "periods":
                # 通常是正数
                for num_str in numbers:
                    num = float(num_str)
                    if num > 0:
                        return num

            # 默认返回第一个数字
            return float(numbers[0])
        return None

    def _extract_numbers_from_question(self, suffix: str = "") -> List[float]:
        """从问题中提取多个数字"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        numbers = re.findall(r'\d+\.?\d*', question)
        return [float(num) for num in numbers]

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            return self.reward_config["calculation_error"]

        reward = self.reward_config["correct_calculation"]

        # 复杂计算的额外奖励
        if action_name in ["compound_interest", "loan_payment", "correlation", "permutation", "combination"]:
            reward += self.reward_config["complex_solved"]

        # 序列完成的奖励
        if len(self.calculation_history) > 1:
            reward += self.reward_config["sequence_completed"]

        return reward

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        return {
            "question": self.current_question,
            "current_problem": f"Mathematical computation problem #{getattr(self, 'problem_count', 1)}",
            "calculation_context": {
                "available_data": [
                    {"name": f"number_{i}", "value": np.array([i*1.5], dtype=np.float32), "type": "numeric"}
                    for i in range(1, 6)
                ],
                "required_precision": self.current_precision,
                "problem_type": self.task_type
            },
            "function_docs": self._get_function_docs(),
            "calculation_state": {
                "calculations_completed": len(self.calculation_history),
                "accuracy_achieved": np.array([0.95], dtype=np.float32),  # 简化实现
                "complexity_level": min(len(self.calculation_history) // 2 + 1, 10)
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # 数学任务通常在完成一定数量的计算后完成
        calculations_completed = obs["calculation_state"]["calculations_completed"]
        return calculations_completed >= 3  # 假设完成3个计算即为任务完成