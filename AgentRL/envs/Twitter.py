from typing import Tuple, Dict, Any

from eval_checker.multi_turn_eval.func_source_code.posting_api import TwitterAPI,DEFAULT_STATE

class TwitterEnv:
    def __init__(self,test_entry: Dict[str, Any]):
        self.twitter_api = TwitterAPI()
        self.test_entry = test_entry
        self._load_scenario_from_test_entry()

    def _load_scenario_from_test_entry(self):
        if "initial_config" in self.test_entry and "TwitterAPI" in self.test_entry["initial_config"]:
            self.twitter_api._load_scenario(self.test_entry["initial_config"]["TwitterAPI"])
        else:
            default_scenario = DEFAULT_STATE
            self.twitter_api._load_scenario(default_scenario)

    def execute_function_call(self, function_name, parameters) -> Tuple[str, bool]:
        try:
            if function_name == "authenticate_twitter":
                result = self.twitter_api.authenticate_twitter(**parameters)
                success = True

            elif function_name == "posting_get_login_status":
                result = self.twitter_api.posting_get_login_status()
                success = True

            elif function_name == "post_tweet":
                result = self.twitter_api.post_tweet(**parameters)
                success = True

            elif function_name == "retweet":
                result = self.twitter_api.retweet(**parameters)
                success = True

            elif function_name == "comment":
                result = self.twitter_api.comment(**parameters)
                success = True

            elif function_name == "mention":
                result = self.twitter_api.mention(**parameters)
                success = True

            elif function_name == "follow_user":
                result = self.twitter_api.follow_user(**parameters)
                success = True

            elif function_name == "list_all_following":
                result = self.twitter_api.list_all_following()
                success = True

            elif function_name == "unfollow_user":
                result = self.twitter_api.unfollow_user(**parameters)
                success = True

            elif function_name == "get_tweet":
                result = self.twitter_api.get_tweet(**parameters)
                success = True

            elif function_name == "get_user_tweets":
                result = self.twitter_api.get_user_tweets(**parameters)
                success = True

            elif function_name == "search_tweets":
                result = self.twitter_api.search_tweets(**parameters)
                success = True

            elif function_name == "get_tweet_comments":
                result = self.twitter_api.get_tweet_comments(**parameters)
                success = True

            elif function_name == "get_user_stats":
                result = self.twitter_api.get_user_stats(**parameters)
                success = True

            else:
                result = f"Function {function_name} not found in TwitterAPI"
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

        return str(result), success
