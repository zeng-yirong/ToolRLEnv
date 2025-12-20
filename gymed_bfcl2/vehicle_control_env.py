"""
车辆控制API环境的Gymnasium适配

基于VehicleControlAPI实现的车辆控制环境，
支持发动机控制、导航、安全系统、气候控制等车辆操作。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json
from copy import deepcopy

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class VehicleControlEnv(FunctionCallingEnv):
    """
    车辆控制环境

    支持以下车辆操作：
    - 发动机控制：start_engine, stop_engine, set_throttle, set_brake
    - 转向控制：steer_left, steer_right, straighten_wheel
    - 导航系统：set_destination, get_location, get_route
    - 安全系统：activate_parking_brake, deactivate_parking_brake, check_safety
    - 气候控制：set_temperature, set_fan_speed, set_ac_mode
    - 车灯控制：turn_on_headlights, turn_off_headlights, set_hazard_lights
    - 信息查询：get_vehicle_status, get_fuel_level, get_battery_voltage
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "vehicle_control",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化车辆控制环境

        Args:
            test_entry: 测试条目，包含车辆控制系统初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入VehicleControlAPI
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.vehicle_control import VehicleControlAPI, DEFAULT_STATE

        self.api = VehicleControlAPI()

        # 默认奖励配置
        default_reward_config = {
            "engine_started": 2.0,          # 发动机启动成功
            "vehicle_moved": 1.5,           # 车辆移动成功
            "destination_reached": 5.0,     # 到达目的地
            "safety_activated": 3.0,        # 安全系统激活
            "navigation_set": 2.0,          # 导航设置成功
            "task_completion": 5.0,         # 任务完成
            "engine_failed": -2.0,          # 发动机启动失败
            "safety_violation": -3.0,       # 安全违规
            "out_of_fuel": -2.5,            # 燃料不足
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)
        # super().__init__(test_entry, max_turns, default_reward_config)
        # 加载车辆控制系统初始状态
        if test_entry and "initial_config" in test_entry:
            if "VehicleControlAPI" in test_entry["initial_config"]:
                self.api._load_scenario(test_entry["initial_config"]["VehicleControlAPI"])

        # 初始化车辆状态
        self.vehicle_actions = []
        self.current_location = "Home"
        self.destination = None

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        actions = [
            "start_engine", "stop_engine", "set_throttle", "set_brake", "steer_left", "steer_right",
            "straighten_wheel", "set_destination", "get_location", "get_route", "activate_parking_brake",
            "deactivate_parking_brake", "check_safety", "set_temperature", "set_fan_speed", "set_ac_mode",
            "turn_on_headlights", "turn_off_headlights", "set_hazard_lights", "get_vehicle_status",
            "get_fuel_level", "get_battery_voltage"
        ]
        return spaces.Discrete(len(actions))

    def _get_observation_space(self) -> spaces.Space:
        """定义观察空间"""
        return spaces.Dict({
            "question": spaces.Text(1000),
            "vehicle_status": spaces.Dict({
                "engine_state": spaces.Text(20),  # stopped/running
                "speed": spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
                "fuel_level": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "battery_voltage": spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32),
                "transmission_gear": spaces.Text(10),
                "odometer": spaces.Box(low=0, high=1e6, shape=(1,), dtype=np.float32)
            }),
            "location_info": spaces.Dict({
                "current_location": spaces.Text(100),
                "destination": spaces.Text(100),
                "distance_to_destination": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
                "estimated_arrival": spaces.Text(50)
            }),
            "safety_status": spaces.Dict({
                "parking_brake": spaces.Text(20),  # engaged/released
                "seat_belts": spaces.Discrete(2),   # 0/1
                "airbags": spaces.Discrete(2),      # 0/1
                "abs_status": spaces.Text(20),      # active/inactive
                "door_status": spaces.Dict({
                    "driver": spaces.Text(10),
                    "passenger": spaces.Text(10),
                    "rear_left": spaces.Text(10),
                    "rear_right": spaces.Text(10)
                })
            }),
            "climate_control": spaces.Dict({
                "temperature": spaces.Box(low=16, high=32, shape=(1,), dtype=np.float32),
                "fan_speed": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "ac_mode": spaces.Text(20),  # auto/cool/heat/off
                "humidity": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
            }),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "driving_state": spaces.Dict({
                "operations_completed": spaces.Discrete(100),
                "distance_traveled": spaces.Box(low=0, high=1e4, shape=(1,), dtype=np.float32),
                "fuel_consumed": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "safety_score": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "start_engine",
                "description": "Start the vehicle engine",
                "parameters": {}
            },
            {
                "name": "stop_engine",
                "description": "Stop the vehicle engine",
                "parameters": {}
            },
            {
                "name": "set_throttle",
                "description": "Set the throttle position",
                "parameters": {
                    "position": "float - throttle position (0-100)"
                }
            },
            {
                "name": "set_brake",
                "description": "Apply brakes",
                "parameters": {
                    "pressure": "float - brake pressure (0-100)"
                }
            },
            {
                "name": "steer_left",
                "description": "Steer the vehicle left",
                "parameters": {
                    "angle": "optional float - steering angle in degrees"
                }
            },
            {
                "name": "steer_right",
                "description": "Steer the vehicle right",
                "parameters": {
                    "angle": "optional float - steering angle in degrees"
                }
            },
            {
                "name": "straighten_wheel",
                "description": "Straighten the steering wheel",
                "parameters": {}
            },
            {
                "name": "set_destination",
                "description": "Set navigation destination",
                "parameters": {
                    "destination": "string - destination address or name"
                }
            },
            {
                "name": "get_location",
                "description": "Get current vehicle location",
                "parameters": {}
            },
            {
                "name": "get_route",
                "description": "Get route to destination",
                "parameters": {}
            },
            {
                "name": "activate_parking_brake",
                "description": "Activate parking brake",
                "parameters": {}
            },
            {
                "name": "deactivate_parking_brake",
                "description": "Deactivate parking brake",
                "parameters": {}
            },
            {
                "name": "check_safety",
                "description": "Check vehicle safety systems",
                "parameters": {}
            },
            {
                "name": "set_temperature",
                "description": "Set climate control temperature",
                "parameters": {
                    "temperature": "float - desired temperature in Celsius"
                }
            },
            {
                "name": "set_fan_speed",
                "description": "Set climate control fan speed",
                "parameters": {
                    "speed": "int - fan speed (0-100)"
                }
            },
            {
                "name": "set_ac_mode",
                "description": "Set air conditioning mode",
                "parameters": {
                    "mode": "string - AC mode (auto/cool/heat/off)"
                }
            },
            {
                "name": "turn_on_headlights",
                "description": "Turn on headlights",
                "parameters": {}
            },
            {
                "name": "turn_off_headlights",
                "description": "Turn off headlights",
                "parameters": {}
            },
            {
                "name": "set_hazard_lights",
                "description": "Set hazard lights on/off",
                "parameters": {
                    "status": "bool - true to turn on, false to turn off"
                }
            },
            {
                "name": "get_vehicle_status",
                "description": "Get comprehensive vehicle status",
                "parameters": {}
            },
            {
                "name": "get_fuel_level",
                "description": "Get current fuel level",
                "parameters": {}
            },
            {
                "name": "get_battery_voltage",
                "description": "Get battery voltage",
                "parameters": {}
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "start_engine", "stop_engine", "set_throttle", "set_brake", "steer_left", "steer_right",
            "straighten_wheel", "set_destination", "get_location", "get_route", "activate_parking_brake",
            "deactivate_parking_brake", "check_safety", "set_temperature", "set_fan_speed", "set_ac_mode",
            "turn_on_headlights", "turn_off_headlights", "set_hazard_lights", "get_vehicle_status",
            "get_fuel_level", "get_battery_voltage"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            if action_name == "start_engine":
                result = {"success": True, "message": "Engine started successfully",
                        "data": {"engine_state": "running", "status": "started"}}
                self.vehicle_actions.append({"action": "start_engine", "timestamp": len(self.vehicle_actions)})

            elif action_name == "stop_engine":
                result = {"success": True, "message": "Engine stopped successfully",
                        "data": {"engine_state": "stopped", "status": "stopped"}}
                self.vehicle_actions.append({"action": "stop_engine", "timestamp": len(self.vehicle_actions)})

            elif action_name == "set_throttle":
                position = self._extract_number_from_question("position") or 50
                result = {"success": True, "message": f"Throttle set to {position}%",
                        "data": {"throttle_position": position}}
                self.vehicle_actions.append({"action": "set_throttle", "position": position, "timestamp": len(self.vehicle_actions)})

            elif action_name == "set_brake":
                pressure = self._extract_number_from_question("pressure") or 30
                result = {"success": True, "message": f"Brake pressure set to {pressure}%",
                        "data": {"brake_pressure": pressure}}

            elif action_name == "steer_left":
                angle = self._extract_number_from_question("angle") or 15
                result = {"success": True, "message": f"Steering left {angle} degrees",
                        "data": {"steering_angle": -angle, "direction": "left"}}

            elif action_name == "steer_right":
                angle = self._extract_number_from_question("angle") or 15
                result = {"success": True, "message": f"Steering right {angle} degrees",
                        "data": {"steering_angle": angle, "direction": "right"}}

            elif action_name == "straighten_wheel":
                result = {"success": True, "message": "Steering wheel straightened",
                        "data": {"steering_angle": 0, "direction": "straight"}}

            elif action_name == "set_destination":
                destination = self._extract_destination_from_question()
                if destination:
                    self.destination = destination
                    result = {"success": True, "message": f"Destination set to {destination}",
                            "data": {"destination": destination, "distance": 15.2, "eta": "25 minutes"}}
                else:
                    result = {"success": False, "message": "Please provide a destination", "data": None}

            elif action_name == "get_location":
                result = {"success": True, "message": f"Current location: {self.current_location}",
                        "data": {"location": self.current_location, "coordinates": "40.7128, -74.0060"}}

            elif action_name == "get_route":
                if self.destination:
                    result = {"success": True, "message": f"Route to {self.destination} calculated",
                            "data": {
                                "destination": self.destination,
                                "distance": 15.2,
                                "estimated_time": "25 minutes",
                                "route": ["Head north on Main St", "Turn right on Oak Ave", "Destination on left"]
                            }}
                else:
                    result = {"success": False, "message": "No destination set", "data": None}

            elif action_name == "activate_parking_brake":
                result = {"success": True, "message": "Parking brake activated",
                        "data": {"parking_brake": "engaged"}}

            elif action_name == "deactivate_parking_brake":
                result = {"success": True, "message": "Parking brake deactivated",
                        "data": {"parking_brake": "released"}}

            elif action_name == "check_safety":
                result = {"success": True, "message": "Safety systems check completed",
                        "data": {
                            "seat_belts": "fastened",
                            "airbags": "ready",
                            "abs": "active",
                            "brake_system": "operational",
                            "safety_score": 95
                        }}

            elif action_name == "set_temperature":
                temperature = self._extract_number_from_question("temperature") or 22
                result = {"success": True, "message": f"Temperature set to {temperature}°C",
                        "data": {"temperature": temperature}}

            elif action_name == "set_fan_speed":
                speed = int(self._extract_number_from_question("speed") or 50)
                result = {"success": True, "message": f"Fan speed set to {speed}%",
                        "data": {"fan_speed": speed}}

            elif action_name == "set_ac_mode":
                mode = self._extract_ac_mode_from_question()
                if mode:
                    result = {"success": True, "message": f"AC mode set to {mode}",
                            "data": {"ac_mode": mode}}
                else:
                    result = {"success": False, "message": "Please specify AC mode (auto/cool/heat/off)", "data": None}

            elif action_name == "turn_on_headlights":
                result = {"success": True, "message": "Headlights turned on",
                        "data": {"headlights": "on"}}

            elif action_name == "turn_off_headlights":
                result = {"success": True, "message": "Headlights turned off",
                        "data": {"headlights": "off"}}

            elif action_name == "set_hazard_lights":
                status = self._extract_boolean_from_question("status")
                if status is not None:
                    status_text = "on" if status else "off"
                    result = {"success": True, "message": f"Hazard lights turned {status_text}",
                            "data": {"hazard_lights": status_text}}
                else:
                    result = {"success": False, "message": "Please specify hazard light status", "data": None}

            elif action_name == "get_vehicle_status":
                result = {"success": True, "message": "Vehicle status retrieved",
                        "data": {
                            "engine": "running",
                            "speed": 0,
                            "fuel": 75.5,
                            "battery": 12.6,
                            "mileage": 45230,
                            "temperature": 90,
                            "oil_pressure": "normal"
                        }}

            elif action_name == "get_fuel_level":
                fuel_level = 75.5  # 简化值
                result = {"success": True, "message": f"Fuel level: {fuel_level}%",
                        "data": {"fuel_level": fuel_level, "range": 320}}

            elif action_name == "get_battery_voltage":
                voltage = 12.6  # 简化值
                result = {"success": True, "message": f"Battery voltage: {voltage}V",
                        "data": {"voltage": voltage, "status": "good"}}

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["engine_failed"]

    def _extract_number_from_question(self, param_name: str) -> Optional[float]:
        """从问题中提取数字"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        numbers = re.findall(r'\d+\.?\d*', question)
        if numbers:
            return float(numbers[0])
        return None

    def _extract_destination_from_question(self) -> Optional[str]:
        """从问题中提取目的地"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 常见的目的地
        known_destinations = ["home", "work", "office", "school", "hospital", "airport", "mall", "restaurant"]

        import re
        # 提取引号中的内容
        matches = re.findall(r'["\']([^"\']+)["\']', question)
        if matches:
            return matches[0]

        # 查找已知目的地
        for dest in known_destinations:
            if dest in question.lower():
                return dest.capitalize()

        return None

    def _extract_ac_mode_from_question(self) -> Optional[str]:
        """从问题中提取空调模式"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        modes = ["auto", "cool", "heat", "off"]
        for mode in modes:
            if mode in question.lower():
                return mode

        return None

    def _extract_boolean_from_question(self, param_name: str) -> Optional[bool]:
        """从问题中提取布尔值"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        if "on" in question.lower() or "true" in question.lower():
            return True
        elif "off" in question.lower() or "false" in question.lower():
            return False

        return None

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            return self.reward_config["engine_failed"]

        # 基础操作奖励
        reward_map = {
            "start_engine": self.reward_config["engine_started"],
            "set_destination": self.reward_config["navigation_set"],
            "activate_parking_brake": self.reward_config["safety_activated"],
            "check_safety": self.reward_config["safety_activated"],
            "set_throttle": self.reward_config["vehicle_moved"],
            "steer_left": self.reward_config["vehicle_moved"],
            "steer_right": self.reward_config["vehicle_moved"],
            "set_temperature": 1.0,
            "set_fan_speed": 1.0,
            "set_ac_mode": 1.0,
            "turn_on_headlights": 1.0,
            "turn_off_headlights": 1.0,
            "get_vehicle_status": 0.5,
            "get_fuel_level": 0.5,
            "get_battery_voltage": 0.5
        }

        return reward_map.get(action_name, 0.5)

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        return {
            "question": self.current_question,
            "vehicle_status": {
                "engine_state": "running",
                "speed": np.array([0.0], dtype=np.float32),
                "fuel_level": np.array([75.5], dtype=np.float32),
                "battery_voltage": np.array([12.6], dtype=np.float32),
                "transmission_gear": "Park",
                "odometer": np.array([45230.5], dtype=np.float32)
            },
            "location_info": {
                "current_location": self.current_location,
                "destination": self.destination or "None",
                "distance_to_destination": np.array([15.2], dtype=np.float32),
                "estimated_arrival": "25 minutes"
            },
            "safety_status": {
                "parking_brake": "released",
                "seat_belts": 1,
                "airbags": 1,
                "abs_status": "active",
                "door_status": {
                    "driver": "locked",
                    "passenger": "locked",
                    "rear_left": "locked",
                    "rear_right": "locked"
                }
            },
            "climate_control": {
                "temperature": np.array([22.0], dtype=np.float32),
                "fan_speed": np.array([50.0], dtype=np.float32),
                "ac_mode": "auto",
                "humidity": np.array([45.0], dtype=np.float32)
            },
            "function_docs": self._get_function_docs(),
            "driving_state": {
                "operations_completed": len(self.vehicle_actions),
                "distance_traveled": np.array([0.0], dtype=np.float32),
                "fuel_consumed": np.array([0.0], dtype=np.float32),
                "safety_score": np.array([95.0], dtype=np.float32)
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # 车辆控制任务通常需要完成一定数量的操作
        driving_state = obs["driving_state"]
        return driving_state["operations_completed"] >= 4  # 至少完成4个车辆操作