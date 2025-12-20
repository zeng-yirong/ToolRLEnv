"""
Twitter API环境的Gymnasium适配

基于TwitterAPI实现的社交媒体操作环境，
支持推文发布、评论、转发、关注用户等社交媒体功能。
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from gymnasium import spaces
import json
from copy import deepcopy

from base_env import FunctionCallingEnv
from utils import FunctionCallExecutor, StateManager


class TwitterAPIEnv(FunctionCallingEnv):
    """
    社交媒体操作环境

    支持以下Twitter操作：
    - 认证管理：login, logout
    - 推文操作：post_tweet, delete_tweet, get_tweet
    - 互动操作：comment, retweet, like_tweet, unlike_tweet
    - 用户管理：follow_user, unfollow_user, get_following_list
    - 搜索操作：search_tweets, get_trending_topics
    """

    def __init__(
        self,
        test_entry: Optional[Dict] = None,
        max_turns: int = 10,
        task_type: str = "social_media",
        reward_config: Optional[Dict] = None
    ):
        """
        初始化Twitter API环境

        Args:
            test_entry: 测试条目，包含Twitter API初始配置
            max_turns: 最大交互轮数
            task_type: 任务类型
            reward_config: 自定义奖励配置
        """
        # 导入TwitterAPI
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'eval_checker', 'multi_turn_eval', 'func_source_code'))
        from eval_checker.multi_turn_eval.func_source_code.posting_api import TwitterAPI, DEFAULT_STATE

        self.api = TwitterAPI()

        # 默认奖励配置
        default_reward_config = {
            "tweet_posted": 3.0,            # 成功发布推文
            "comment_added": 2.0,           # 成功添加评论
            "retweeted": 2.5,               # 成功转发
            "user_followed": 1.5,           # 成功关注用户
            "tweet_liked": 1.0,             # 成功点赞推文
            "search_completed": 1.5,        # 完成搜索
            "task_completion": 5.0,         # 任务完成
            "post_failed": -2.0,            # 发布失败
            "not_authenticated": -2.5,     # 未认证
            "user_not_found": -1.5,         # 用户未找到
            "tweet_not_found": -1.0,        # 推文未找到
        }

        if reward_config:
            default_reward_config.update(reward_config)

        super().__init__(test_entry, max_turns, task_type, default_reward_config)

        # 加载Twitter API初始状态
        if test_entry and "initial_config" in test_entry:
            if "TwitterAPI" in test_entry["initial_config"]:
                self.api._load_scenario(test_entry["initial_config"]["TwitterAPI"])

        # 初始化Twitter状态
        self.current_user = None
        self.tweet_history = []
        self.social_interactions = []

    def _get_action_space(self) -> spaces.Space:
        """定义动作空间"""
        # 基于TwitterAPI的实际函数
        actions = [
            "login", "logout", "post_tweet", "delete_tweet", "get_tweet",
            "comment", "retweet", "like_tweet", "unlike_tweet",
            "follow_user", "unfollow_user", "get_following_list",
            "search_tweets", "get_trending_topics"
        ]
        return spaces.Discrete(len(actions))

    def _get_observation_space(self) -> spaces.Space:
        """定义观察空间"""
        return spaces.Dict({
            "question": spaces.Text(1000),
            "current_user": spaces.Text(100),
            "authentication_state": spaces.Dict({
                "is_authenticated": spaces.Discrete(2),
                "username": spaces.Text(100),
                "followers_count": spaces.Discrete(10000),
                "following_count": spaces.Discrete(1000)
            }),
            "twitter_feed": spaces.Sequence(spaces.Dict({
                "tweet_id": spaces.Text(50),
                "author": spaces.Text(100),
                "content": spaces.Text(280),
                "timestamp": spaces.Text(50),
                "likes": spaces.Discrete(1000),
                "retweets": spaces.Discrete(500),
                "comments": spaces.Discrete(200)
            })),
            "social_stats": spaces.Dict({
                "tweets_posted": spaces.Discrete(100),
                "comments_made": spaces.Discrete(200),
                "retweets_made": spaces.Discrete(50),
                "users_followed": spaces.Discrete(100),
                "likes_given": spaces.Discrete(500)
            }),
            "function_docs": spaces.Sequence(spaces.Dict({
                "name": spaces.Text(100),
                "description": spaces.Text(500),
                "parameters": spaces.Dict()
            })),
            "twitter_state": spaces.Dict({
                "character_limit": spaces.Discrete(280),
                "current_trends": spaces.Sequence(spaces.Text(100)),
                "mentions": spaces.Sequence(spaces.Text(100)),
                "notifications": spaces.Discrete(50)
            })
        })

    def _get_function_docs(self) -> List[Dict]:
        """获取函数文档"""
        return [
            {
                "name": "login",
                "description": "Log in to Twitter with username and password",
                "parameters": {
                    "username": "string - Twitter username",
                    "password": "string - Twitter password"
                }
            },
            {
                "name": "logout",
                "description": "Log out from Twitter",
                "parameters": {}
            },
            {
                "name": "post_tweet",
                "description": "Post a new tweet",
                "parameters": {
                    "content": "string - tweet content (max 280 characters)"
                }
            },
            {
                "name": "delete_tweet",
                "description": "Delete a tweet",
                "parameters": {
                    "tweet_id": "string - ID of the tweet to delete"
                }
            },
            {
                "name": "get_tweet",
                "description": "Get details of a specific tweet",
                "parameters": {
                    "tweet_id": "string - ID of the tweet"
                }
            },
            {
                "name": "comment",
                "description": "Add a comment to a tweet",
                "parameters": {
                    "tweet_id": "string - ID of the tweet to comment on",
                    "comment": "string - comment content"
                }
            },
            {
                "name": "retweet",
                "description": "Retweet a tweet",
                "parameters": {
                    "tweet_id": "string - ID of the tweet to retweet"
                }
            },
            {
                "name": "like_tweet",
                "description": "Like a tweet",
                "parameters": {
                    "tweet_id": "string - ID of the tweet to like"
                }
            },
            {
                "name": "unlike_tweet",
                "description": "Unlike a tweet",
                "parameters": {
                    "tweet_id": "string - ID of the tweet to unlike"
                }
            },
            {
                "name": "follow_user",
                "description": "Follow a user",
                "parameters": {
                    "username": "string - username of the user to follow"
                }
            },
            {
                "name": "unfollow_user",
                "description": "Unfollow a user",
                "parameters": {
                    "username": "string - username of the user to unfollow"
                }
            },
            {
                "name": "get_following_list",
                "description": "Get the list of users you are following",
                "parameters": {}
            },
            {
                "name": "search_tweets",
                "description": "Search for tweets with keywords",
                "parameters": {
                    "query": "string - search query",
                    "max_results": "optional int - maximum number of results"
                }
            },
            {
                "name": "get_trending_topics",
                "description": "Get current trending topics",
                "parameters": {
                    "location": "optional string - location for trending topics"
                }
            }
        ]

    def _execute_action(self, action: int, obs: Dict, info: Dict) -> Tuple[bool, Dict, float]:
        """执行动作"""
        action_names = [
            "login", "logout", "post_tweet", "delete_tweet", "get_tweet",
            "comment", "retweet", "like_tweet", "unlike_tweet",
            "follow_user", "unfollow_user", "get_following_list",
            "search_tweets", "get_trending_topics"
        ]

        action_name = action_names[action]
        result = {"success": False, "message": "", "data": None}

        try:
            if action_name == "login":
                username = self._extract_username_from_question()
                password = self._extract_password_from_question()
                if username and password:
                    # 简化的登录验证
                    if (username == "john" and password == "john123") or \
                       (username == "tech_guru" and password == "securePass123") or \
                       (username == "analyst_pro" and password == "Kj8#mP9$vL2"):
                        self.current_user = username
                        result = {"success": True, "message": f"Successfully logged in as {username}",
                                "data": {"username": username, "authenticated": True}}
                    else:
                        result = {"success": False, "message": "Invalid username or password", "data": None}

            elif action_name == "logout":
                result = {"success": True, "message": "Successfully logged out", "data": {"action": "logout"}}
                self.current_user = None

            elif action_name == "post_tweet":
                if self.current_user:
                    content = self._extract_tweet_content_from_question()
                    if content and len(content) <= 280:
                        tweet_id = f"tweet_{len(self.tweet_history) + 1}"
                        self.tweet_history.append({
                            "tweet_id": tweet_id,
                            "author": self.current_user,
                            "content": content,
                            "timestamp": "2024-01-01 12:00:00",
                            "likes": 0,
                            "retweets": 0,
                            "comments": 0
                        })
                        result = {"success": True, "message": f"Tweet posted successfully: {content[:50]}...",
                                "data": {"tweet_id": tweet_id, "content": content}}
                    else:
                        result = {"success": False, "message": "Tweet content too long or empty", "data": None}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "delete_tweet":
                if self.current_user:
                    tweet_id = self._extract_tweet_id_from_question()
                    if tweet_id:
                        # 简化的删除实现
                        result = {"success": True, "message": f"Tweet {tweet_id} deleted successfully",
                                "data": {"tweet_id": tweet_id, "action": "deleted"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_tweet":
                tweet_id = self._extract_tweet_id_from_question()
                if tweet_id:
                    # 简化的获取推文实现
                    sample_tweet = {
                        "tweet_id": tweet_id,
                        "author": "sample_user",
                        "content": f"This is sample content for tweet {tweet_id}",
                        "timestamp": "2024-01-01 12:00:00",
                        "likes": 10,
                        "retweets": 5,
                        "comments": 2
                    }
                    result = {"success": True, "message": f"Tweet details for {tweet_id}", "data": sample_tweet}

            elif action_name == "comment":
                if self.current_user:
                    tweet_id = self._extract_tweet_id_from_question()
                    comment_text = self._extract_comment_text_from_question()
                    if tweet_id and comment_text:
                        self.social_interactions.append({
                            "action": "comment",
                            "user": self.current_user,
                            "tweet_id": tweet_id,
                            "comment": comment_text,
                            "timestamp": len(self.social_interactions)
                        })
                        result = {"success": True, "message": f"Comment added to tweet {tweet_id}",
                                "data": {"tweet_id": tweet_id, "comment": comment_text}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "retweet":
                if self.current_user:
                    tweet_id = self._extract_tweet_id_from_question()
                    if tweet_id:
                        self.social_interactions.append({
                            "action": "retweet",
                            "user": self.current_user,
                            "tweet_id": tweet_id,
                            "timestamp": len(self.social_interactions)
                        })
                        result = {"success": True, "message": f"Retweeted tweet {tweet_id}",
                                "data": {"tweet_id": tweet_id, "action": "retweeted"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "like_tweet":
                if self.current_user:
                    tweet_id = self._extract_tweet_id_from_question()
                    if tweet_id:
                        self.social_interactions.append({
                            "action": "like",
                            "user": self.current_user,
                            "tweet_id": tweet_id,
                            "timestamp": len(self.social_interactions)
                        })
                        result = {"success": True, "message": f"Liked tweet {tweet_id}",
                                "data": {"tweet_id": tweet_id, "action": "liked"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "unlike_tweet":
                if self.current_user:
                    tweet_id = self._extract_tweet_id_from_question()
                    if tweet_id:
                        result = {"success": True, "message": f"Unliked tweet {tweet_id}",
                                "data": {"tweet_id": tweet_id, "action": "unliked"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "follow_user":
                if self.current_user:
                    username = self._extract_username_from_question()
                    if username and username != self.current_user:
                        result = {"success": True, "message": f"Now following {username}",
                                "data": {"username": username, "action": "followed"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "unfollow_user":
                if self.current_user:
                    username = self._extract_username_from_question()
                    if username:
                        result = {"success": True, "message": f"Unfollowed {username}",
                                "data": {"username": username, "action": "unfollowed"}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "get_following_list":
                if self.current_user:
                    # 简化的关注列表实现
                    following_list = ["alice", "bob", "tech_innovator", "future_visionary"]
                    result = {"success": True, "message": f"Users you follow: {', '.join(following_list)}",
                            "data": {"following": following_list}}
                else:
                    result = {"success": False, "message": "Please log in first", "data": None}

            elif action_name == "search_tweets":
                query = self._extract_search_query_from_question()
                if query:
                    # 简化的搜索实现
                    results = [
                        {"tweet_id": "search_1", "author": "user1", "content": f"Tweet about {query} #1"},
                        {"tweet_id": "search_2", "author": "user2", "content": f"Another tweet about {query} #2"}
                    ]
                    result = {"success": True, "message": f"Found {len(results)} tweets about '{query}'",
                            "data": {"query": query, "results": results}}
                else:
                    result = {"success": False, "message": "Please provide a search query", "data": None}

            elif action_name == "get_trending_topics":
                location = self._extract_location_from_question()
                # 简化的热门话题实现
                trending = ["#AI", "#Technology", "#Innovation", "#MachineLearning", "#DataScience"]
                result = {"success": True, "message": f"Trending topics in {location or 'Global'}: {', '.join(trending)}",
                        "data": {"location": location or "Global", "trends": trending}}

            # 计算奖励
            reward = self._calculate_reward(result, action_name)
            success = result["success"]

            return success, result, reward

        except Exception as e:
            error_result = {"success": False, "message": f"Error: {str(e)}", "data": None}
            return False, error_result, self.reward_config["post_failed"]

    def _extract_username_from_question(self) -> Optional[str]:
        """从问题中提取用户名"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 常见的用户名模式
        known_usernames = ["john", "tech_guru", "analyst_pro", "alice", "bob"]

        import re
        # 查找用户名模式
        for username in known_usernames:
            if username in question.lower():
                return username

        # 查找 @username 模式
        username_pattern = r'@(\w+)'
        matches = re.findall(username_pattern, question)
        if matches:
            return matches[0]

        return None

    def _extract_password_from_question(self) -> Optional[str]:
        """从问题中提取密码"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 简化的密码提取（在实际应用中不应该这样做）
        known_passwords = ["john123", "securePass123", "Kj8#mP9$vL2"]

        # 在示例问题中查找密码模式
        import re
        password_pattern = r'password["\']?\s*[:is]?\s*["\']?([^"\'\s]+)["\']?'
        matches = re.findall(password_pattern, question, re.IGNORECASE)
        if matches:
            return matches[0]

        return None

    def _extract_tweet_content_from_question(self) -> Optional[str]:
        """从问题中提取推文内容"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 提取引号中的内容
        import re
        matches = re.findall(r'["\']([^"\']+)["\']', question)
        if matches:
            return matches[0]

        # 查找常见推文模式
        if "post" in question.lower() or "share" in question.lower():
            # 简单的内容提取
            words = question.split()
            content_words = []
            skip_words = ["post", "share", "tweet", "a", "the", "as", "with", "and", "or", "but"]

            for word in words:
                if word.lower() not in skip_words and len(word) > 2:
                    content_words.append(word)
                    if len(" ".join(content_words)) > 200:  # 限制长度
                        break

            if content_words:
                return " ".join(content_words)

        return None

    def _extract_tweet_id_from_question(self) -> Optional[str]:
        """从问题中提取推文ID"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 匹配推文ID模式
        patterns = [
            r'tweet["\']?\s*(?:id)?\s*[:#]?\s*(\w+)',
            r'(\w+)(?:\s+tweet)',
            r'tweet["\']?\s*[:#]?\s*(\w+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _extract_comment_text_from_question(self) -> Optional[str]:
        """从问题中提取评论内容"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 提取引号中的内容
        import re
        matches = re.findall(r'["\']([^"\']+)["\']', question)
        if matches:
            return matches[0]

        # 查找"comment"关键词后的内容
        if "comment" in question.lower():
            words = question.lower().split()
            comment_index = -1
            for i, word in enumerate(words):
                if "comment" in word:
                    comment_index = i
                    break

            if comment_index >= 0 and comment_index + 1 < len(words):
                return " ".join(words[comment_index + 1:comment_index + 6])  # 取后面的5个词

        return None

    def _extract_search_query_from_question(self) -> Optional[str]:
        """从问题中提取搜索查询"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        import re
        # 查找搜索关键词模式
        patterns = [
            r'search for ["\']([^"\']+)["\']',
            r'search["\']?\s+(?:for\s+)?["\']?([^"\'\s]+)["\']?',
            r'about["\']?\s+["\']?([^"\'\s]+)["\']?'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def _extract_location_from_question(self) -> Optional[str]:
        """从问题中提取位置"""
        question = self.current_observation.get("question", "") if hasattr(self, 'current_observation') else ""

        # 常见的位置
        known_locations = ["global", "us", "uk", "japan", "china", "india", "france", "germany"]

        for location in known_locations:
            if location in question.lower():
                return location

        return None

    def _calculate_reward(self, result: Dict, action_name: str) -> float:
        """计算奖励"""
        if not result["success"]:
            if "log in" in result["message"].lower():
                return self.reward_config["not_authenticated"]
            elif "not found" in result["message"].lower():
                return self.reward_config["user_not_found"]
            else:
                return self.reward_config["post_failed"]

        # 基础操作奖励
        reward_map = {
            "post_tweet": self.reward_config["tweet_posted"],
            "comment": self.reward_config["comment_added"],
            "retweet": self.reward_config["retweeted"],
            "follow_user": self.reward_config["user_followed"],
            "like_tweet": self.reward_config["tweet_liked"],
            "search_tweets": self.reward_config["search_completed"],
            "get_trending_topics": self.reward_config["search_completed"],
            "login": 1.0,
            "delete_tweet": 1.5,
            "get_tweet": 0.5,
            "get_following_list": 0.5
        }

        return reward_map.get(action_name, 0.5)

    def get_observation(self, info: Dict) -> Dict:
        """获取当前观察"""
        # 简化的推文feed
        sample_feed = [
            {
                "tweet_id": "sample_1",
                "author": "tech_innovator",
                "content": "Excited about the future of AI! #AI #Technology",
                "timestamp": "2024-01-01 10:00:00",
                "likes": 42,
                "retweets": 12,
                "comments": 8
            },
            {
                "tweet_id": "sample_2",
                "author": "data_scientist",
                "content": "Just published a new paper on machine learning #ML #Research",
                "timestamp": "2024-01-01 11:30:00",
                "likes": 28,
                "retweets": 6,
                "comments": 3
            }
        ]

        return {
            "question": self.current_question,
            "current_user": self.current_user or "None",
            "authentication_state": {
                "is_authenticated": 1 if self.current_user else 0,
                "username": self.current_user or "",
                "followers_count": 150 if self.current_user else 0,
                "following_count": 50 if self.current_user else 0
            },
            "twitter_feed": sample_feed,
            "social_stats": {
                "tweets_posted": len(self.tweet_history),
                "comments_made": len([i for i in self.social_interactions if i["action"] == "comment"]),
                "retweets_made": len([i for i in self.social_interactions if i["action"] == "retweet"]),
                "users_followed": len([i for i in self.social_interactions if i["action"] == "follow"]),
                "likes_given": len([i for i in self.social_interactions if i["action"] == "like"])
            },
            "function_docs": self._get_function_docs(),
            "twitter_state": {
                "character_limit": 280,
                "current_trends": ["#AI", "#Technology", "#Innovation"],
                "mentions": ["@tech_innovator"],
                "notifications": 2 if self.current_user else 0
            }
        }

    def _check_task_completion(self, obs: Dict, info: Dict) -> bool:
        """检查任务是否完成"""
        # Twitter任务通常需要完成一定数量的社交媒体互动
        social_stats = obs["social_stats"]
        total_interactions = (social_stats["tweets_posted"] +
                            social_stats["comments_made"] +
                            social_stats["retweets_made"] +
                            social_stats["likes_given"])
        return total_interactions >= 3  # 至少完成3个社交媒体操作