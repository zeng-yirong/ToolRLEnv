# Tool RL Environments (TRE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)]()

## 📖 项目介绍 (Introduction)

**Tool RL Environments** 是一个专为强化学习（Reinforcement Learning, RL）和大型语言模型（LLM）代理设计的训练环境套件。

本项目旨在通过提供多样化、交互式的工具环境，评估和提升智能体在 **工具调用（Function Calling）** 和 **API 交互** 方面的能力。本项目的设计理念类似于 Berkeley Function Calling Leaderboard (BFCL)，但更侧重于强化学习训练过程中的交互反馈。

## 🚀 环境列表 (Environments)

本项目包含多个领域的仿真环境，涵盖了从系统操作、社交媒体到物理控制的多种场景。所有环境均基于标准化的工具接口设计：

### 🖥️ 系统与工具 (System & Utilities)
* **File System**: 一个属于文件系统的工具。这是一个简单的文件系统，允许用户执行基本的文件操作，如导航目录、创建文件和目录、读取和写入文件等。
* **Math**: 属于数学 API，提供各种数学运算功能。
* **Web Search**: 属于 Web 搜索 API 类别。它提供搜索网络和浏览搜索结果的功能。

### 🧠 记忆管理 (Memory Suite)
该套件提供了一系列 API 来测试代理的长期记忆和信息检索能力：
* **Memory_kv**: 属于内存套件，提供 API 以与基于键值（Key-Value）的内存系统进行交互。
* **Memory_rec_sum**: 属于内存套件，提供通过递归摘要（Recursive Summarization）来管理内存数据的 API。
* **Memory_vector**: 属于内存套件，提供 API 以与基于键值的内存系统进行交互。

### 📱 社交与通讯 (Communication & Social)
* **Twitter**: 属于 TwitterAPI，提供发布推文、转发、评论和关注 Twitter 用户等核心功能 [cite: 6]。
* **Message**: 属于消息 API，用于管理工作区中的用户交互。

### 🚗 物理控制 (Physical Control)
* **Vehicle**: 属于车辆控制系统，允许用户控制汽车的各个方面，如引擎、车门、气候控制、灯光等。

## 🛠️ 安装 (Installation)

```bash
git clone https://github.com/zeng-yirong/ToolRLEnv.git
cd AgentRL
pip install -r requirements.txt
