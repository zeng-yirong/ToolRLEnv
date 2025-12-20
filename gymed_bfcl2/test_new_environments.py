import traceback

from general_env import GeneralEnv

test_entry = {
        "id": "debug_test_entry",
        "question": [[{"role": "user", "content": "Please_verify_my_travel_information_and_set_a_budget_limit"}]],
        ##### 需要执行的函数
        "function": [
            {
                "description": "verified_user",  #### 函数描述
                "name": "authenticate_travel",   #### 函数名，需要与源文件中一致
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "client_id",'properties_value':{"type": "string",'description':'111',"value":""}},    ####函数参数，properties_key为参数名，properties_value[type]为参数类型，properties_value[value]为参数值
                        {'properties_key': "client_secret", 'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "refresh_token",'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "grant_type",'properties_value': {"type": "string", 'description': '111',"value":"read"}},
                        {'properties_key': "user_first_name", 'properties_value': {"type": "string", 'description': '111',"value":"Shen"}},
                        {'properties_key': "user_last_name",'properties_value': {"type": "string", 'description': '111',"value":"You"}}
                    ]),
                    "required": tuple(["client_id", "client_secret", "refresh_token", "grant_type", "user_first_name", "user_last_name"]),
                    "env":"travel"
                }
            },
            {
                "description": "fill_fuel_tank",
                "name": "fill_fuel_tank",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "fuelAmount", 'properties_value': {"type": "float", 'description': 'fuelAmount',"value":"3.0"}},
                        # {'properties_key': "budget_limit",'properties_value': {"type": "number", 'description': '222',"value":"40000"}},
                    ]),
                    "required": tuple(["fuelAmount"]),
                    "env":"vehicle_control"
                }
            },
            {
                "description": "displayCarStatus",
                "name": "display_car_status",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "option", 'properties_value': {"type": "string", 'description': 'option',"value":"fuel"}},
                        # {'properties_key': "budget_limit",'properties_value': {"type": "number", 'description': '222',"value":"40000"}},
                    ]),
                    "required": tuple(["access_token", "budget_limit"]),
                    "env":"vehicle_control"
                }
            },
            {
                "description": "fetch_url_content",
                "name": "fetch_url_content",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "url", 'properties_value': {"type": "string", 'description': 'url',"value":"http://baidu.com"}},
                        {'properties_key': "mode",'properties_value': {"type": "string", 'description': 'mode',"value":"raw"}},
                    ]),
                    "required": tuple(["url", "mode"]),
                    "env":"web_search"
                }
            },
            {
                "description": "authenticate_twitter",
                "name": "authenticate_twitter",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "username", 'properties_value': {"type": "string", 'description': 'username',"value":"john"}},
                        {'properties_key': "password",'properties_value': {"type": "string", 'description': 'password',"value":"john123"}},
                    ]),
                    "required": tuple(["username", "password"]),
                    "env":"twitter"
                }
            },
            {
                "description": "get_current_time",
                "name": "get_current_time",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        # {'properties_key': "username", 'properties_value': {"type": "string", 'description': 'username',"value":"john"}},
                        # {'properties_key': "password",'properties_value': {"type": "string", 'description': 'password',"value":"john123"}},
                    ]),
                    "required": tuple([]),
                    "env":"trading_bot"
                }
            },
            {
                "description": "create_ticket",
                "name": "create_ticket",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "title", 'properties_value': {"type": "string", 'description': 'title',"value":"my_ticket"}},
                        {'properties_key': "description",'properties_value': {"type": "string", 'description': 'description',"value":"my_ticket"}},
                        {'properties_key': "priority",'properties_value': {"type": "int", 'description': 'priority',"value":"1"}},
                    ]),
                    "required": tuple(["title", "description", "priority"]),
                    "env":"ticket"
                }
            },
            {
                "description": "list_users",
                "name": "list_users",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        # {'properties_key': "username", 'properties_value': {"type": "string", 'description': 'username',"value":"john"}},
                        # {'properties_key': "password",'properties_value': {"type": "string", 'description': 'password',"value":"john123"}},
                    ]),
                    "required": tuple([]),
                    "env":"message"
                }
            },
            {
                "description": "add",
                "name": "add",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "a", 'properties_value': {"type": "float", 'description': 'a',"value":"2.7"}},
                        {'properties_key': "b",'properties_value': {"type": "float", 'description': 'b',"value":"3.5"}},
                    ]),
                    "required": tuple(["a","b"]),
                    "env":"math"
                }
            },
            {
                "description": "pwd",
                "name": "pwd",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        # {'properties_key': "a", 'properties_value': {"type": "float", 'description': 'a',"value":"2.7"}},
                        # {'properties_key': "b",'properties_value': {"type": "float", 'description': 'b',"value":"3.5"}},
                    ]),
                    "required": tuple([]),
                    "env":"gorilla"
                }
            },
            {
                "description": "mkdir",
                "name": "mkdir",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "dir_name", 'properties_value': {"type": "string", 'description': 'dir_name',"value":"test"}},
                        # {'properties_key': "b",'properties_value': {"type": "float", 'description': 'b',"value":"3.5"}},
                    ]),
                    "required": tuple(["dir_name"]),
                    "env":"gorilla"
                }
            },
            {
                "description": "cd",
                "name": "cd",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "folder", 'properties_value': {"type": "string", 'description': 'folder',"value":"test"}},
                        # {'properties_key': "b",'properties_value': {"type": "float", 'description': 'b',"value":"3.5"}},
                    ]),
                    "required": tuple(["folder"]),
                    "env":"gorilla"
                }
            },
        ],
    #### 各个环境的初始配置
        "initial_config": {
            "TravelAPI": {
                "credit_card_list": {},
                "booking_record": {},
                "access_token": "123",
                "token_type": None,
                "token_expires_in": None,
                "token_scope": None,
                "user_first_name": None,
                "user_last_name": None,
                "budget_limit": 5000.0
            },
            "VehicleControlAPI":{
                "random_seed": 141053,
                "fuelLevel": 1.0,
                "batteryVoltage": 12.6,
                "engine_state": "stopped",
                "remainingUnlockedDoors": 4,
                "doorStatus": {
                    "driver": "unlocked",
                    "passenger": "unlocked",
                    "rear_left": "unlocked",
                    "rear_right": "unlocked",
                },
                "acTemperature": 25.0,
                "fanSpeed": 50,
                "acMode": "auto",
                "humidityLevel": 50.0,
                "headLightStatus": "off",
                "parkingBrakeStatus": "released",
                "_parkingBrakeForce": 0.0,
                "_slopeAngle": 0.0,
                "brakePedalStatus": "released",
                "brakePedalForce": 0.0,
                "distanceToNextVehicle": 50.0,
                "cruiseStatus": "inactive",
                "destination": "None",
                "frontLeftTirePressure": 32.0,
                "frontRightTirePressure": 32.0,
                "rearLeftTirePressure": 30.0,
                "rearRightTirePressure": 30.0,
            },
            "TicketAPI":{
                "ticket_queue": [],
                "ticket_counter": 1,
                "current_user": "YouShen",
            }
        },
        "path": ["TravelAPI.authenticate_travel", "TravelAPI.set_budget_limit"],
        "involved_classes": ["TravelAPI","VehicleControlAPI"]
    }

def main():
    env = GeneralEnv(test_entry = test_entry)   #### test_entry为要运行的数据
    # print("使用 gymnasium.utils.env_checker 验证环境")
    # try:
    #     from gymnasium.utils.env_checker import check_env
    #     # obs,inf = env.reset()
    #     # print(obs)
    #     check_env(env)
    #     print("✓ 环境检查通过！")
    #
    # except Exception as e:
    #     print(f"✗ 环境检查失败: {e}")
    #     traceback.print_exc()
    # 测试数据
    env.reset()
    for step in range(12):
        print(step) ### step为test_entry[function]中各个要执行函数的索引值
        obs, reward, terminated, truncated, info = env.step(step)

if __name__ == "__main__":
    main()