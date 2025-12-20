from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.math_api import MathAPI

class MathEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.math_api = MathAPI()
        self.reward_config = {
            "correct_calculation": 2.0,     # 正确计算
            "complex_solved": 3.0,          # 解决复杂问题
            "sequence_completed": 1.5,      # 完成计算序列
            "precision_achieved": 2.0,      # 达到精度要求
            "task_completion": 5.0,         # 任务完成
            "calculation_error": -2.0,      # 计算错误
            "invalid_input": -1.5,          # 无效输入
            "precision_lost": -1.0,         # 精度丢失
        }
        self.calculation_history = []
        self.current_precision = 5  # 默认精度
        self.test_entry = test_entry

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "logarithm":
                result = self.math_api.logarithm(**parameters)
                success = True

            elif function_name == "mean":
                result = self.math_api.mean(**parameters)
                success = True

            elif function_name == "standard_deviation":
                result = self.math_api.standard_deviation(**parameters)
                success = True

            elif function_name == "si_unit_conversion":
                result = self.math_api.si_unit_conversion(**parameters)
                success = True

            elif function_name == "imperial_si_conversion":
                result = self.math_api.imperial_si_conversion(**parameters)
                success = True

            elif function_name == "add":
                result = self.math_api.add(**parameters)
                success = True

            elif function_name == "subtract":
                result = self.math_api.subtract(**parameters)
                success = True

            elif function_name == "multiply":
                result = self.math_api.multiply(**parameters)
                success = True

            elif function_name == "divide":
                result = self.math_api.divide(**parameters)
                success = True

            elif function_name == "power":
                result = self.math_api.power(**parameters)
                success = True

            elif function_name == "square_root":
                result = self.math_api.square_root(**parameters)
                success = True

            elif function_name == "absolute_value":
                result = self.math_api.absolute_value(**parameters)
                success = True

            elif function_name == "round_number":
                result = self.math_api.round_number(**parameters)
                success = True

            elif function_name == "percentage":
                result = self.math_api.percentage(**parameters)
                success = True

            elif function_name == "min_value":
                result = self.math_api.min_value(**parameters)
                success = True

            elif function_name == "max_value":
                result = self.math_api.max_value(**parameters)
                success = True

            elif function_name == "sum_values":
                result = self.math_api.sum_values(**parameters)
                success = True

            else:
                result = f"Function {function_name} not found in MathAPI"
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

        self.calculation_history.append(operation)

        return str(result), success