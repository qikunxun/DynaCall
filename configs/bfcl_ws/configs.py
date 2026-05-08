from configs.bfcl_ws.gpt_prompts import OUTPUT_PROMPT as GPT_OUTPUT_PROMPT
from configs.bfcl_ws.gpt_prompts import PLANNER_PROMPT as GPT_PLANNER_PROMPT
try:
    from configs.bfcl_ws.gpt_prompts import (
        PLANNER_PROMPT_REPLAN as GPT_PLANNER_PROMPT_REPLAN,
    )
except ImportError:
    GPT_PLANNER_PROMPT_REPLAN = GPT_PLANNER_PROMPT

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
