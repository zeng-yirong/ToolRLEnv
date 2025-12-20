"""
网络搜索API环境的Gymnasium适配

基于WebSearchAPI实现的网络搜索环境，
支持搜索引擎查询、网页浏览、搜索结果获取等功能。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json
from copy import deepcopy

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class WebSearchEnv(FunctionCallingEnv):
    """
    网络搜索环境

    支持以下搜索操作：
    - 搜索查询：search_engine_query, web_search, image_search
    - 网页浏览：browse_page, get_page_content, extract_links
    - 结果处理：get_search_results, filter_results, sort_results
    - 高级搜索：news_search, academic_search, video_search
    - 工具功能：save_result, export_results, get_search_history
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "web_search",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化网络搜索环境

        Args:
            test_entry: 测试条目，包含搜索系统初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入WebSearchAPI
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI

        self.api = WebSearchAPI()

        # 默认奖励配置
        default_reward_config = {
            "search_completed": 2.0,        # 搜索完成
            "relevant_found": 3.0,          # 找到相关结果
            "page_browsed": 1.5,            # 网页浏览成功
            "information_extracted": 2.5,   # 信息提取成功
            "results_filtered": 1.0,        # 结果过滤成功
            "task_completion": 5.0,         # 任务完成
            "search_failed": -2.0,          # 搜索失败
            "no_results": -1.5,             # 无搜索结果
            "access_denied": -2.5,          # 访问被拒绝
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)
        # 加载搜索系统初始状态
        if test_entry and "initial_config" in test_entry:
            if "WebSearch" in test_entry["initial_config"]:
                self.api._load_scenario(test_entry["initial_config"]["WebSearch"])

        # 初始化搜索状态
        self.search_history = []
        self.browsed_pages = []
        self.saved_results = []

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        actions = [
            "search_engine_query", "web_search", "image_search", "browse_page", "get_page_content",
            "extract_links", "get_search_results", "filter_results", "sort_results", "news_search",
            "academic_search", "video_search", "save_result", "export_results", "get_search_history"
        ]
        return spaces.Discrete(len(actions))

    def _get_observation_space(self) -> spaces.Space:
        """定义观察空间"""
        return spaces.Dict({
            "question": spaces.Text(1000),
            "search_results": spaces.Sequence(spaces.Dict({
                "title": spaces.Text(200),
                "url": spaces.Text(500),
                "snippet": spaces.Text(300),
                "relevance_score": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "source": spaces.Text(100)
            })),
            "current_page": spaces.Dict({
                "url": spaces.Text(500),
                "title": spaces.Text(200),
                "content": spaces.Text(2000),
                "word_count": spaces.Discrete(10000)
            }),
            "search_history": spaces.Sequence(spaces.Dict({
                "query": spaces.Text(200),
                "timestamp": spaces.Text(50),
                "result_count": spaces.Discrete(100)
            })),
            "search_stats": spaces.Dict({
                "total_searches": spaces.Discrete(100),
                "pages_browsed": spaces.Discrete(50),
                "results_saved": spaces.Discrete(200),
                "success_rate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            }),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "search_state": spaces.Dict({
                "current_query": spaces.Text(200),
                "total_results": spaces.Discrete(1000),
                "page_number": spaces.Discrete(10),
                "search_mode": spaces.Text(50)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "search_engine_query",
                "description": "Perform a search engine query",
                "parameters": {
                    "keywords": "string - search keywords",
                    "max_results": "optional int - maximum number of results",
                    "region": "optional string - search region",
                    "language": "optional string - search language"
                }
            },
            {
                "name": "web_search",
                "description": "General web search",
                "parameters": {
                    "query": "string - search query",
                    "safesearch": "optional bool - enable safe search"
                }
            },
            {
                "name": "image_search",
                "description": "Search for images",
                "parameters": {
                    "query": "string - image search query",
                    "image_type": "optional string - type of images",
                    "size": "optional string - image size"
                }
            },
            {
                "name": "browse_page",
                "description": "Browse a specific webpage",
                "parameters": {
                    "url": "string - URL of the page to browse"
                }
            },
            {
                "name": "get_page_content",
                "description": "Get the content of a web page",
                "parameters": {
                    "url": "string - URL of the page",
                    "extract_text": "optional bool - extract text only"
                }
            },
            {
                "name": "extract_links",
                "description": "Extract links from a web page",
                "parameters": {
                    "url": "string - URL of the page",
                    "link_type": "optional string - type of links to extract"
                }
            },
            {
                "name": "get_search_results",
                "description": "Get search results for a query",
                "parameters": {
                    "query": "string - search query",
                    "start_index": "optional int - starting index",
                    "count": "optional int - number of results"
                }
            },
            {
                "name": "filter_results",
                "description": "Filter search results",
                "parameters": {
                    "results": "list - search results to filter",
                    "criteria": "string - filtering criteria",
                    "value": "string - filter value"
                }
            },
            {
                "name": "sort_results",
                "description": "Sort search results",
                "parameters": {
                    "results": "list - search results to sort",
                    "sort_by": "string - sorting criteria",
                    "order": "optional string - sort order (asc/desc)"
                }
            },
            {
                "name": "news_search",
                "description": "Search for news articles",
                "parameters": {
                    "query": "string - news search query",
                    "time_range": "optional string - time range"
                }
            },
            {
                "name": "academic_search",
                "description": "Search for academic papers",
                "parameters": {
                    "query": "string - academic search query",
                    "field": "optional string - academic field"
                }
            },
            {
                "name": "video_search",
                "description": "Search for videos",
                "parameters": {
                    "query": "string - video search query",
                    "duration": "optional string - video duration",
                    "quality": "optional string - video quality"
                }
            },
            {
                "name": "save_result",
                "description": "Save a search result",
                "parameters": {
                    "result": "dict - search result to save",
                    "tags": "optional list - tags for the saved result"
                }
            },
            {
                "name": "export_results",
                "description": "Export search results",
                "parameters": {
                    "results": "list - results to export",
                    "format": "string - export format (json/csv/txt)"
                }
            },
            {
                "name": "get_search_history",
                "description": "Get search history",
                "parameters": {
                    "limit": "optional int - number of recent searches"
                }
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "search_engine_query", "web_search", "image_search", "browse_page", "get_page_content",
            "extract_links", "get_search_results", "filter_results", "sort_results", "news_search",
            "academic_search", "video_search", "save_result", "export_results", "get_search_history"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            if action_name == "search_engine_query":
                keywords = self._extract_search_query_from_question()
                max_results = self._extract_number_from_question("max_results") or 10
                region = self._extract_region_from_question()
                language = self._extract_language_from_question()

                if keywords:
                    # 简化的搜索实现
                    search_results = [
                        {
                            "title": f"Search result 1 for '{keywords}'",
                            "url": f"https://example.com/result1?q={keywords}",
                            "snippet": f"This is a relevant snippet about {keywords}",
                            "relevance_score": 0.95,
                            "source": "web"
                        },
                        {
                            "title": f"Search result 2 for '{keywords}'",
                            "url": f"https://example.com/result2?q={keywords}",
                            "snippet": f"Another relevant result about {keywords}",
                            "relevance_score": 0.88,
                            "source": "web"
                        }
                    ]

                    self.search_history.append({
                        "query": keywords,
                        "timestamp": len(self.search_history),
                        "result_count": len(search_results)
                    })

                    result = {"success": True, "message": f"Found {len(search_results)} results for '{keywords}'",
                            "data": {"query": keywords, "results": search_results, "total_count": len(search_results)}}
                else:
                    result = {"success": False, "message": "Please provide search keywords", "data": None}

            elif action_name == "web_search":
                query = self._extract_search_query_from_question()
                if query:
                    # 简化的网络搜索实现
                    results = [
                        {"title": f"Web result: {query}", "url": "https://example.com/web1", "snippet": f"Information about {query}"}
                    ]
                    result = {"success": True, "message": f"Web search for '{query}' completed",
                            "data": {"query": query, "results": results}}
                else:
                    result = {"success": False, "message": "Please provide a search query", "data": None}

            elif action_name == "image_search":
                query = self._extract_search_query_from_question()
                if query:
                    results = [
                        {"title": f"Image result: {query}", "url": "https://example.com/image1.jpg", "size": "800x600"}
                    ]
                    result = {"success": True, "message": f"Found images for '{query}'",
                            "data": {"query": query, "results": results}}
                else:
                    result = {"success": False, "message": "Please provide image search query", "data": None}

            elif action_name == "browse_page":
                url = self._extract_url_from_question()
                if url:
                    self.browsed_pages.append({
                        "url": url,
                        "timestamp": len(self.browsed_pages)
                    })
                    result = {"success": True, "message": f"Browsing page: {url}",
                            "data": {"url": url, "title": f"Page Title for {url}", "content": f"Sample content from {url}"}}
                else:
                    result = {"success": False, "message": "Please provide a valid URL", "data": None}

            elif action_name == "get_page_content":
                url = self._extract_url_from_question()
                if url:
                    content = f"This is the main content extracted from {url}. It contains relevant information about the topic."
                    result = {"success": True, "message": f"Content extracted from {url}",
                            "data": {"url": url, "content": content, "word_count": len(content.split())}}
                else:
                    result = {"success": False, "message": "Please provide a valid URL", "data": None}

            elif action_name == "extract_links":
                url = self._extract_url_from_question()
                if url:
                    links = [
                        {"url": "https://example.com/link1", "text": "Related Article 1"},
                        {"url": "https://example.com/link2", "text": "Related Article 2"},
                        {"url": "https://example.com/link3", "text": "Download Link"}
                    ]
                    result = {"success": True, "message": f"Extracted {len(links)} links from {url}",
                            "data": {"url": url, "links": links}}
                else:
                    result = {"success": False, "message": "Please provide a valid URL", "data": None}

            elif action_name == "get_search_results":
                query = self._extract_search_query_from_question()
                start_index = self._extract_number_from_question("start_index") or 0
                count = self._extract_number_from_question("count") or 10

                if query:
                    results = [
                        {"title": f"Result {i+1} for {query}", "url": f"https://example.com/result{i+1}"}
                        for i in range(min(count, 5))
                    ]
                    result = {"success": True, "message": f"Retrieved {len(results)} results for '{query}'",
                            "data": {"query": query, "results": results, "start_index": start_index}}
                else:
                    result = {"success": False, "message": "Please provide a search query", "data": None}

            elif action_name == "filter_results":
                criteria = self._extract_filter_criteria_from_question()
                value = self._extract_filter_value_from_question()
                if criteria and value:
                    # 简化的过滤实现
                    filtered_results = [
                        {"title": f"Filtered result 1", "url": "https://example.com/filtered1"}
                    ]
                    result = {"success": True, "message": f"Filtered results by {criteria}: {value}",
                            "data": {"criteria": criteria, "value": value, "results": filtered_results}}
                else:
                    result = {"success": False, "message": "Please provide filter criteria and value", "data": None}

            elif action_name == "sort_results":
                sort_by = self._extract_sort_criteria_from_question()
                if sort_by:
                    result = {"success": True, "message": f"Results sorted by {sort_by}",
                            "data": {"sort_by": sort_by, "order": "desc"}}
                else:
                    result = {"success": False, "message": "Please provide sorting criteria", "data": None}

            elif action_name == "news_search":
                query = self._extract_search_query_from_question()
                if query:
                    results = [
                        {"title": f"Breaking News: {query}", "url": "https://news.example.com/article1", "date": "2024-01-01"}
                    ]
                    result = {"success": True, "message": f"Found news about '{query}'",
                            "data": {"query": query, "results": results}}
                else:
                    result = {"success": False, "message": "Please provide news search query", "data": None}

            elif action_name == "academic_search":
                query = self._extract_search_query_from_question()
                if query:
                    results = [
                        {"title": f"Academic Paper: {query}", "authors": ["Author 1", "Author 2"], "journal": "Science Journal"}
                    ]
                    result = {"success": True, "message": f"Found academic papers about '{query}'",
                            "data": {"query": query, "results": results}}
                else:
                    result = {"success": False, "message": "Please provide academic search query", "data": None}

            elif action_name == "video_search":
                query = self._extract_search_query_from_question()
                if query:
                    results = [
                        {"title": f"Video: {query}", "url": "https://video.example.com/video1", "duration": "5:23"}
                    ]
                    result = {"success": True, "message": f"Found videos about '{query}'",
                            "data": {"query": query, "results": results}}
                else:
                    result = {"success": False, "message": "Please provide video search query", "data": None}

            elif action_name == "save_result":
                # 简化的保存结果实现
                saved_result = {"title": "Saved Result", "url": "https://example.com/saved"}
                self.saved_results.append(saved_result)
                result = {"success": True, "message": "Result saved successfully",
                        "data": {"saved_result": saved_result, "total_saved": len(self.saved_results)}}

            elif action_name == "export_results":
                format_type = self._extract_export_format_from_question()
                if format_type:
                    result = {"success": True, "message": f"Results exported in {format_type} format",
                            "data": {"format": format_type, "export_count": len(self.saved_results)}}
                else:
                    result = {"success": False, "message": "Please specify export format", "data": None}

            elif action_name == "get_search_history":
                limit = self._extract_number_from_question("limit") or 10
                history = self.search_history[-limit:] if self.search_history else []
                result = {"success": True, "message": f"Retrieved {len(history)} recent searches",
                        "data": {"history": history, "total_searches": len(self.search_history)}}

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["search_failed"]

    def _extract_search_query_from_question(self) -> Optional[str]:
        """从问题中提取搜索查询"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 查找搜索关键词模式
        patterns = [
            r'search for ["\']([^"\']+)["\']',
            r'search["\']?\s+(?:for\s+)?["\']?([^"\'\s]+)["\']?',
            r'about["\']?\s+["\']?([^"\'\s]+)["\']?',
            r'look up ["\']([^"\']+)["\']',
            r'find information about ["\']([^"\']+)["\']'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return matches[0]

        # 提取引号中的内容
        matches = re.findall(r'["\']([^"\']+)["\']', question)
        if matches:
            return matches[0]

        return None

    def _extract_url_from_question(self) -> Optional[str]:
        """从问题中提取URL"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 匹配URL模式
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        matches = re.findall(url_pattern, question)
        if matches:
            return matches[0]

        return None

    def _extract_number_from_question(self, param_name: str) -> Optional[int]:
        """从问题中提取数字"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        numbers = re.findall(r'\d+', question)
        if numbers:
            return int(numbers[0])
        return None

    def _extract_region_from_question(self) -> Optional[str]:
        """从问题中提取地区"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        regions = ["us", "uk", "canada", "australia", "global", "international"]
        for region in regions:
            if region in question.lower():
                return region

        return None

    def _extract_language_from_question(self) -> Optional[str]:
        """从问题中提取语言"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        languages = ["english", "spanish", "french", "german", "chinese", "japanese"]
        for language in languages:
            if language in question.lower():
                return language

        return None

    def _extract_filter_criteria_from_question(self) -> Optional[str]:
        """从问题中提取过滤条件"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        criteria = ["date", "relevance", "source", "type", "domain"]
        for criterion in criteria:
            if criterion in question.lower():
                return criterion

        return None

    def _extract_filter_value_from_question(self) -> Optional[str]:
        """从问题中提取过滤值"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 查找过滤值模式
        patterns = [
            r'filter by ["\']([^"\']+)["\']',
            r'["\']([^"\']+)["\']\s+filter',
            r'value["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _extract_sort_criteria_from_question(self) -> Optional[str]:
        """从问题中提取排序条件"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        criteria = ["date", "relevance", "popularity", "rating", "alphabetical"]
        for criterion in criteria:
            if f"sort by {criterion}" in question.lower() or criterion in question.lower():
                return criterion

        return None

    def _extract_export_format_from_question(self) -> Optional[str]:
        """从问题中提取导出格式"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        formats = ["json", "csv", "txt", "xml", "pdf"]
        for format_type in formats:
            if format_type in question.lower():
                return format_type

        return None

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            if "no results" in result["message"].lower():
                return self.reward_config["no_results"]
            elif "access" in result["message"].lower():
                return self.reward_config["access_denied"]
            else:
                return self.reward_config["search_failed"]

        # 基础操作奖励
        reward_map = {
            "search_engine_query": self.reward_config["search_completed"],
            "web_search": self.reward_config["search_completed"],
            "image_search": self.reward_config["search_completed"],
            "news_search": self.reward_config["search_completed"],
            "academic_search": self.reward_config["search_completed"],
            "video_search": self.reward_config["search_completed"],
            "browse_page": self.reward_config["page_browsed"],
            "get_page_content": self.reward_config["information_extracted"],
            "extract_links": self.reward_config["information_extracted"],
            "get_search_results": self.reward_config["relevant_found"],
            "filter_results": self.reward_config["results_filtered"],
            "sort_results": self.reward_config["results_filtered"],
            "save_result": 1.5,
            "export_results": 2.0,
            "get_search_history": 0.5
        }

        return reward_map.get(action_name, 0.5)

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        # 简化的搜索结果
        sample_results = [
            {
                "title": "Search Result 1",
                "url": "https://example.com/result1",
                "snippet": "This is a relevant search result snippet",
                "relevance_score": np.array([0.95], dtype=np.float32),
                "source": "web"
            },
            {
                "title": "Search Result 2",
                "url": "https://example.com/result2",
                "snippet": "Another relevant search result",
                "relevance_score": np.array([0.88], dtype=np.float32),
                "source": "web"
            }
        ]

        # 简化的搜索历史
        search_history = [
            {"query": "machine learning", "timestamp": "2024-01-01 10:00", "result_count": 10},
            {"query": "AI research", "timestamp": "2024-01-01 10:30", "result_count": 15}
        ]

        return {
            "question": self.current_question,
            "search_results": sample_results,
            "current_page": {
                "url": "https://example.com/current",
                "title": "Current Page Title",
                "content": "This is the content of the current web page being viewed.",
                "word_count": 42
            },
            "search_history": search_history,
            "search_stats": {
                "total_searches": len(self.search_history),
                "pages_browsed": len(self.browsed_pages),
                "results_saved": len(self.saved_results),
                "success_rate": np.array([0.92], dtype=np.float32)
            },
            "function_docs": self._get_function_docs(),
            "search_state": {
                "current_query": self.search_history[-1]["query"] if self.search_history else "",
                "total_results": 25,
                "page_number": 1,
                "search_mode": "web"
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # 搜索任务通常需要完成一定数量的搜索和浏览操作
        search_stats = obs["search_stats"]
        total_operations = search_stats["total_searches"] + search_stats["pages_browsed"]
        return total_operations >= 3  # 至少完成3个搜索相关操作