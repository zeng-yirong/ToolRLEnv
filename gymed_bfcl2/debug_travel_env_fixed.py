#!/usr/bin/env python3
"""
TravelEnv 环境检测脚本 (修复版)

该脚本使用多种策略检测和调试 TravelBookingEnv 环境，确保其正常工作。
修复了需要 test_entry 参数的问题。
"""
import sys
import traceback
import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple

def load_travel_test_entry():
    """加载包含 TravelAPI 的测试条目"""
    print("正在加载 TravelAPI 测试数据...")

    # 检查数据文件是否存在
    data_files = [
        "data/travel_booking.json"
    ]

    for data_file in data_files:
        if not os.path.exists(data_file):
            print(f"⚠ 数据文件不存在: {data_file}")
            continue

        print(f"正在搜索文件: {data_file}")
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num > 1000:  # 限制搜索行数
                        break
                    try:
                        test_entry = json.loads(line.strip())
                        # 检查是否包含 TravelAPI
                        if test_entry.get("involved_classes") and "TravelAPI" in test_entry["involved_classes"]:
                            print(f"✓ 找到 TravelAPI 测试条目: {test_entry['id']}")
                            return test_entry
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"读取文件 {data_file} 时出错: {e}")
            continue

    # 如果没找到，创建一个简单的测试条目
    print("⚠ 未找到 TravelAPI 测试条目，创建默认测试条目")
    return create_default_test_entry()

def create_default_test_entry():
    """创建默认的 TravelAPI 测试条目"""
    return {
        "id": "debug_test_entry",
        "question": [[{"role": "user", "content": "Please_verify_my_travel_information_and_set_a_budget_limit"}]],
        "function": [
            {
                "description": "verified_user",
                "name": "authenticate_travel",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "client_id",'properties_value':{"type": "string",'description':'111',"value":""}},
                        {'properties_key': "client_secret", 'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "refresh_token",'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "grant_type",'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "user_first_name", 'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "user_last_name",'properties_value': {"type": "string", 'description': '111',"value":""}}
                    ]),
                    "required": tuple(["client_id", "client_secret", "refresh_token", "grant_type", "user_first_name", "user_last_name"])
                }
            },
            {
                "description": "Set_budget_constraints",
                "name": "set_budget_limit",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "access_token", 'properties_value': {"type": "string", 'description': '222',"value":""}},
                        {'properties_key': "budget_limit",'properties_value': {"type": "number", 'description': '222',"value":""}},
                    ]),
                    "required": tuple(["access_token", "budget_limit"])
                }
            }
        ],
        "initial_config": {
            "TravelAPI": {
                "credit_card_list": {},
                "booking_record": {},
                "access_token": None,
                "token_type": None,
                "token_expires_in": None,
                "token_scope": None,
                "user_first_name": None,
                "user_last_name": None,
                "budget_limit": 5000.0
            }
        },
        "path": ["TravelAPI.authenticate_travel", "TravelAPI.set_budget_limit"],
        "involved_classes": ["TravelAPI"]
    }

def test_travel_import():
    """测试基本导入和初始化"""
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)

    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from travel_env import TravelBookingEnv
        print("✓ 成功导入 TravelBookingEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = TravelBookingEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_web_search():
    """测试基本导入和初始化"""
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)

    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from web_search_env import  WebSearchEnv
        print("✓ 成功导入 WebSearchEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = WebSearchEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_vehicle_control():
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)
    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from vehicle_control_env import VehicleControlEnv
        print("✓ 成功导入 VehicleControlEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = VehicleControlEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_twitter_api():
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)
    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from twitter_api_env import TwitterAPIEnv
        print("✓ 成功导入 TwitterAPIEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = TwitterAPIEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_trading_bot():
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)
    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from trading_bot_env import TradingBotEnv
        print("✓ 成功导入 TradingBotEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = TradingBotEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_ticket_api():
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)
    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from ticket_api_env import TicketAPIEnv
        print("✓ 成功导入 TicketAPIEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = TicketAPIEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_message_api():
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)
    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from message_api_env import MessageAPIEnv
        print("✓ 成功导入 MessageAPIEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = MessageAPIEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_math_api():
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)
    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from math_api_env import MathAPIEnv
        print("✓ 成功导入 MathAPIEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = MathAPIEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_gorilla_file_system():
    print("=" * 60)
    print("1. 测试基本导入和初始化")
    print("=" * 60)
    try:
        # 测试导入
        print("正在导入 gymed_bfcl...")
        from gorilla_file_system_env import GorillaFileSystemEnv
        print("✓ 成功导入 GorillaFileSystemEnv")

        # 加载测试数据
        test_entry = load_travel_test_entry()
        if test_entry is None:
            print("✗ 无法加载测试数据")
            return None

        # 测试基本初始化
        print("正在创建环境实例...")
        env = GorillaFileSystemEnv(test_entry=test_entry, max_turns=10)
        print("✓ 成功创建环境实例")
        print(f"环境类型: {type(env)}")
        print(f"环境模块: {env.__module__}")
        print(f"测试条目ID: {test_entry['id']}")
        print(f"涉及API类: {test_entry['involved_classes']}")

        return env

    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        traceback.print_exc()
        return None

def test_env_checker(env):
    """使用 gymnasium.utils.env_checker 验证环境"""
    print("\n" + "=" * 60)
    print("2. 使用 gymnasium.utils.env_checker 验证环境")
    print("=" * 60)

    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env)
        print("✓ 环境检查通过！")

    except Exception as e:
        print(f"✗ 环境检查失败: {e}")
        traceback.print_exc()

        # 尝试识别具体问题
        print("\n正在尝试识别具体问题...")
        try:
            # 检查基本属性
            print(f"动作空间: {env.action_space}")
            print(f"观察空间: {env.observation_space}")

            # 尝试重置
            obs, info = env.reset(seed=42)
            print("✓ 重置功能正常")

            # 尝试单步执行
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print("✓ 单步执行功能正常")

        except Exception as e2:
            print(f"✗ 基本功能测试失败: {e2}")

def test_observation_space_structure(env):
    """检查观察空间结构"""
    print("\n" + "=" * 60)
    print("3. 检查观察空间结构")
    print("=" * 60)

    try:
        obs, info = env.reset(seed=42)

        print("观察空间内容:")
        for key, value in obs.items():
            if isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value).__name__} (长度: {len(value)})")
                if key == "function_docs" and len(value) > 0:
                    print(f"    - 第一个函数文档键: {list(value[0].keys()) if value[0] else 'None'}")
                    print(f"    - 函数数量: {len(value)}")
                elif key == "available_airports" and len(value) > 0:
                    print(f"    - 示例机场: {value[:3]}...")
            elif isinstance(value, dict):
                print(f"  {key}: {type(value).__name__} (键: {list(value.keys())})")
                if key == "authentication_state":
                    print(f"    - 认证状态: {value.get('is_authenticated', 'Unknown')}")
                elif key == "booking_state":
                    print(f"    - 预订状态: {value}")
                elif key == "financial_status":
                    print(f"    - 财务状态: {value}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")

        print(f"\n观察空间键总数: {len(obs)}")
        print("✓ 观察空间结构检查完成")

    except Exception as e:
        print(f"✗ 观察空间检查失败: {e}")
        traceback.print_exc()

def test_action_space_and_rewards(env):
    """验证动作空间和奖励机制"""
    print("\n" + "=" * 60)
    print("4. 验证动作空间和奖励机制")
    print("=" * 60)

    try:
        obs, info = env.reset(seed=42)

        print(f"动作空间: {env.action_space}")
        print(f"动作空间类型: {type(env.action_space)}")
        print(f"动作数量: {env.action_space.n}")

        # 测试每个动作
        rewards = []
        for action in range(min(env.action_space.n, 5)):  # 限制测试前5个动作
            obs_current, reward, terminated, truncated, info_current = env.step(action)
            rewards.append(reward)

            print(f"动作 {action}: 奖励 = {reward:.2f}, 终止 = {terminated}, 截断 = {truncated}")

            if terminated or truncated:
                print(f"  -> 环境在动作 {action} 后结束")
                break

        print(f"\n所有动作的奖励范围: [{min(rewards):.2f}, {max(rewards):.2f}]")
        print("✓ 动作空间和奖励机制检查完成")

    except Exception as e:
        print(f"✗ 动作空间和奖励机制检查失败: {e}")
        traceback.print_exc()

def test_manual_action_sequence(env):
    """手动测试特定动作序列"""
    print("\n" + "=" * 60)
    print("5. 手动测试特定动作序列")
    print("=" * 60)

    try:
        # 使用不同的测试条目
        # test_entries = []
        #
        # # 尝试加载多个 TravelAPI 测试条目
        # for data_file in ["data/BFCL_v4_multi_turn_base.json"]:
        #     if os.path.exists(data_file):
        #         with open(data_file, 'r', encoding='utf-8') as f:
        #             count = 0
        #             for line in f:
        #                 if count >= 3:  # 只取前3个
        #                     break
        #                 try:
        #                     test_entry = json.loads(line.strip())
        #                     if test_entry.get("involved_classes") and "TravelAPI" in test_entry["involved_classes"]:
        #                         test_entries.append(test_entry)
        #                         count += 1
        #                 except json.JSONDecodeError:
        #                     continue
        #
        # if not test_entries:
        #     test_entries = [env.test_entry]  # 使用当前环境的测试条目
        #
        # for i, test_entry in enumerate(test_entries):
        #     print(f"\n--- 测试条目 {i+1}: {test_entry['id']} ---")

            try:
                # 测试数据
                test_entry = {
        "id": "debug_test_entry",
        "question": [[{"role": "user", "content": "Please_verify_my_travel_information_and_set_a_budget_limit"}]],
        "function": [
            {
                "description": "verified_user",
                "name": "authenticate_travel",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "client_id",'properties_value':{"type": "string",'description':'111',"value":""}},
                        {'properties_key': "client_secret", 'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "refresh_token",'properties_value': {"type": "string", 'description': '111',"value":""}},
                        {'properties_key': "grant_type",'properties_value': {"type": "string", 'description': '111',"value":"read"}},
                        {'properties_key': "user_first_name", 'properties_value': {"type": "string", 'description': '111',"value":"Shen"}},
                        {'properties_key': "user_last_name",'properties_value': {"type": "string", 'description': '111',"value":"You"}}
                    ]),
                    "required": tuple(["client_id", "client_secret", "refresh_token", "grant_type", "user_first_name", "user_last_name"])
                }
            },
            {
                "description": "Set_budget_constraints",
                "name": "set_budget_limit",
                "parameters": {
                    "type": "object",
                    "properties": tuple([
                        {'properties_key': "access_token", 'properties_value': {"type": "string", 'description': '222',"value":"123"}},
                        {'properties_key': "budget_limit",'properties_value': {"type": "number", 'description': '222',"value":"40000"}},
                    ]),
                    "required": tuple(["access_token", "budget_limit"])
                }
            }
        ],
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
            }
        },
        "path": ["TravelAPI.authenticate_travel", "TravelAPI.set_budget_limit"],
        "involved_classes": ["TravelAPI"]
    }


                # 创建新环境实例
                from travel_env import TravelBookingEnv
                test_env = TravelBookingEnv(test_entry=test_entry, max_turns=10)
                obs, info = test_env.reset(seed=42)

                print(f"  问题数量: {len(test_entry['question'])}")
                print(f"  函数数量: {len(test_entry['function'])}")
                print(f"  预期路径: {' -> '.join(test_entry['path'])}")

                # 测试合理的动作序列
                total_reward = 0
                for step in (0,1):
                    action = step  # 简单策略：按顺序执行动作
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    total_reward += reward
                    print(f"    步骤 {step}: 动作 {action}, 奖励 {reward:.2f}")
                    if terminated or truncated:
                        print(f"    -> 任务{step}完成/结束")
                #完成一系列动作后设置为True，将不再使用这个环境
                test_env.done = True
                print(f"    总奖励: {total_reward:.2f}")
                print(f"    ✓ 测试条目 {test_entry['id']} 完成")

            except Exception as e:
                print(f"    ✗ 测试条目 {test_entry['id']} 失败: {e}")

    except Exception as e:
        print(f"✗ 手动动作序列测试失败: {e}")
        traceback.print_exc()

def test_reset_functionality(env):
    """测试环境重置功能"""
    print("\n" + "=" * 60)
    print("6. 测试环境重置功能")
    print("=" * 60)

    try:
        # 测试多次重置
        for i in range(3):
            print(f"\n重置测试 {i+1}:")

            obs, info = env.reset(seed=i)

            print(f"  重置后的状态:")
            if isinstance(obs, dict):
                if "question" in obs:
                    question = obs["question"]
                    if isinstance(question, str):
                        print(f"    问题: {question[:50]}...")
                    elif isinstance(question, list) and len(question) > 0:
                        print(f"    问题列表长度: {len(question)}")
                if "function_docs" in obs:
                    print(f"    可用函数数量: {len(obs['function_docs'])}")
            else:
                print(f"    观察类型: {type(obs)}")

            # 执行一个动作
            if hasattr(env, 'action_space') and env.action_space.n > 0:
                action = 0  # 第一个动作
                try:
                    obs_after, reward, terminated, truncated, info_after = env.step(action)
                    print(f"    执行动作 {action} 后: 奖励 {reward:.2f}")
                except Exception as e:
                    print(f"    执行动作失败: {e}")

        print("\n✓ 重置功能测试完成")

    except Exception as e:
        print(f"✗ 重置功能测试失败: {e}")
        traceback.print_exc()

def test_state_management(env):
    """检查状态管理功能"""
    print("\n" + "=" * 60)
    print("7. 检查状态管理功能")
    print("=" * 60)

    try:
        obs, info = env.reset(seed=42)

        print("初始状态检查:")
        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, str) and len(sub_value) > 50:
                            print(f"    {sub_key}: {sub_value[:50]}...")
                        else:
                            print(f"    {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    print(f"  {key}: 列表 (长度: {len(value)})")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"    第一个元素的键: {list(value[0].keys())}")
                else:
                    if isinstance(value, str) and len(value) > 50:
                        print(f"  {key}: {value[:50]}...")
                    else:
                        print(f"  {key}: {value}")
        else:
            print(f"  观察类型: {type(obs)}")

        # 执行几个动作并观察状态变化
        print("\n状态变化测试:")
        for step in range(min(3, env.action_space.n)):
            action = step  # 依次执行前几个动作
            try:
                obs_after, reward, terminated, truncated, info_after = env.step(action)

                print(f"  动作 {action} 后:")
                print(f"    奖励: {reward:.2f}")
                print(f"    终止: {terminated}, 截断: {truncated}")

                if isinstance(obs_after, dict) and "question" in obs_after:
                    print(f"    问题变化: 已更新")

            except Exception as e:
                print(f"    动作 {action} 执行失败: {e}")

        print("\n✓ 状态管理检查完成")

    except Exception as e:
        print(f"✗ 状态管理检查失败: {e}")
        traceback.print_exc()

def test_edge_cases(env):
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("8. 测试边界情况")
    print("=" * 60)

    try:
        # 测试无效动作
        obs, info = env.reset(seed=42)

        print("测试无效动作:")
        try:
            invalid_action = env.action_space.n  # 超出范围的动作
            obs, reward, terminated, truncated, info = env.step(invalid_action)
            print(f"  无效动作 {invalid_action}: 环境处理正常")
        except Exception as e:
            print(f"  无效动作 {invalid_action}: 预期的错误 {type(e).__name__}")

        # 测试最大轮数限制
        print("\n测试最大轮数限制:")
        try:
            # 创建一个轮数较少的环境
            from travel_env import TravelBookingEnv
            env_short = TravelBookingEnv(test_entry=env.test_entry, max_turns=3)
            obs, info = env_short.reset(seed=42)

            steps = 0
            while True:
                action = 0  # 第一个动作
                obs, reward, terminated, truncated, info = env_short.step(action)
                steps += 1

                if terminated or truncated:
                    print(f"  环境在 {steps} 步后结束")
                    if truncated:
                        print("  ✓ 达到最大轮数限制")
                    break
                elif steps >= 10:
                    print("  ⚠ 环境未在合理步数内结束")
                    break

        except Exception as e:
            print(f"  最大轮数测试失败: {e}")

        print("\n✓ 边界情况测试完成")

    except Exception as e:
        print(f"✗ 边界情况测试失败: {e}")
        traceback.print_exc()

def main():
    """主测试函数"""
    print("GorillaFileSystemEnv 环境检测开始 (修复版)")
    print("=" * 60)

    # 1. 基本导入测试
    env = test_gorilla_file_system()
    if env is None:
        print("\n✗ 环境导入失败，终止测试")
        return

    # 2. 环境检查器测试
    test_env_checker(env)

    # 3. 观察空间结构检查
    test_observation_space_structure(env)

    # 4. 动作空间和奖励机制验证
    test_action_space_and_rewards(env)

    # 5. 手动动作序列测试
    test_manual_action_sequence(env)

    # 6. 重置功能测试
    test_reset_functionality(env)

    # 7. 状态管理检查
    test_state_management(env)

    # 8. 边界情况测试
    test_edge_cases(env)

    print("\n" + "=" * 60)
    print("环境检测完成")
    print("=" * 60)

    # 总结
    print("\n总结:")
    print("如果看到多个 ✓ 标记，说明环境基本功能正常。")
    print("如果看到 ✗ 标记，请检查相应的错误信息并修复问题。")
    print("\n常见问题:")
    print("1. 导入错误 - 检查 Python 路径和模块安装")
    print("2. 环境检查失败 - 检查 Gymnasium API 实现")
    print("3. 动作/观察空间问题 - 检查空间定义")
    print("4. 状态管理问题 - 检查状态更新逻辑")
    print("5. 测试数据问题 - 确保数据文件存在且格式正确")

if __name__ == "__main__":
    main()