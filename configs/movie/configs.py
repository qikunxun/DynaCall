from configs.movie.gpt_prompts import OUTPUT_PROMPT as GPT_OUTPUT_PROMPT
from configs.movie.gpt_prompts import PLANNER_PROMPT as GPT_PLANNER_PROMPT
from configs.movie.gpt_prompts import PLANNER_PROMPT_REPLAN as GPT_PLANNER_PROMPT_REPLAN

CONFIGS = {
    "default_model": "gpt-4-1106-preview",
    "prompts": {
        "gpt": {
            "planner_prompt": GPT_PLANNER_PROMPT,
            "planner_prompt_replan": GPT_PLANNER_PROMPT_REPLAN,
            "output_prompt": GPT_OUTPUT_PROMPT,
        },
        "llama": {
            "planner_prompt": GPT_PLANNER_PROMPT,
            "planner_prompt_replan": GPT_PLANNER_PROMPT_REPLAN,
            "output_prompt": GPT_OUTPUT_PROMPT,
        },
    },
    "max_replans": 3,
}
