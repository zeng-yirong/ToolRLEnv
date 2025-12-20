from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.vehicle_control import VehicleControlAPI,DEFAULT_STATE

class VehicleControlEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.vehicle_api = VehicleControlAPI()
        self.reward_config = {
            "engine_started": 2.0,  # 发动机启动成功
            "vehicle_moved": 1.5,  # 车辆移动成功
            "destination_reached": 5.0,  # 到达目的地
            "safety_activated": 3.0,  # 安全系统激活
            "navigation_set": 2.0,  # 导航设置成功
            "task_completion": 5.0,  # 任务完成
            "engine_failed": -2.0,  # 发动机启动失败
            "safety_violation": -3.0,  # 安全违规
            "out_of_fuel": -2.5,  # 燃料不足
        }
        self.vehicle_actions = []
        self.current_location = "Home"
        self.destination = None
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        """从传入的 test_entry 加载场景到 TravelAPI"""
        # 从 test_entry 中获取 initial_config 并加载
        if "initial_config" in self.test_entry and "VehicleControlAPI" in self.test_entry["initial_config"]:
            # 直接传递配置到 TravelAPI
            self.vehicle_api._load_scenario(self.test_entry["initial_config"]["VehicleControlAPI"])
        else:
            default_scenario = DEFAULT_STATE
            self.vehicle_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "start_engine":
                result = self.vehicle_api.startEngine(**parameters)
                success = True

            elif function_name == "fill_fuel_tank":
                result = self.vehicle_api.fillFuelTank(**parameters)
                success = True

            elif function_name == "lock_doors":
                result = self.vehicle_api.lockDoors(**parameters)
                success = True

            elif function_name == "get_outside_temperature_from_google":
                result = self.vehicle_api.get_outside_temperature_from_google()
                success = True

            elif function_name == "set_head_lights":
                result = self.vehicle_api.setHeadlights(**parameters)
                success = True

            elif function_name == "display_car_status":
                result = self.vehicle_api.displayCarStatus(**parameters)
                success = True

            elif function_name == "activate_parking_brake":
                result = self.vehicle_api.activateParkingBrake(**parameters)
                success = True

            elif function_name == "press_brake_pedal":
                result = self.vehicle_api.pressBrakePedal(**parameters)
                success = True

            elif function_name == "release_brake_pedal":
                result = self.vehicle_api.releaseBrakePedal()
                success = True

            elif function_name == "set_cruise_control":
                result = self.vehicle_api.setCruiseControl(**parameters)
                success = True

            elif function_name == "get_current_speed":
                result = self.vehicle_api.get_current_speed()
                success = True

            elif function_name == "display_log":
                result = self.vehicle_api.display_log(**parameters)
                success = True

            elif function_name == "estimate_drive_feasibility_by_mileage":
                result = self.vehicle_api.estimate_drive_feasibility_by_mileage(**parameters)
                success = True

            elif function_name == "liter_to_gallon":
                result = self.vehicle_api.liter_to_gallon(**parameters)
                success = True

            elif function_name == "estimate_distance":
                result = self.vehicle_api.estimate_distance(**parameters)
                success = True

            elif function_name == "get_zipcode_based_on_city":
                result = self.vehicle_api.get_zipcode_based_on_city(**parameters)
                success = True

            elif function_name == "set_navigation":
                result = self.vehicle_api.set_navigation(**parameters)
                success = True

            elif function_name == "check_tire_pressure":
                result = self.vehicle_api.check_tire_pressure()
                success = True

            elif function_name == "find_nearest_tire_shop":
                result = self.vehicle_api.find_nearest_tire_shop()
                success = True

            else:
                result = f"Function {function_name} not found in VehicleControlAPI"
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

        self.vehicle_actions.append(operation)

        return str(result), success