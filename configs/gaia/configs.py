from configs.gaia.gpt_prompts import OUTPUT_PROMPT as GPT_OUTPUT_PROMPT
from configs.gaia.gpt_prompts import OUTPUT_PROMPT_FINAL as GPT_OUTPUT_PROMPT_FINAL
from configs.gaia.gpt_prompts import PLANNER_CRITIC_PROMPT as GPT_PLANNER_CRITIC_PROMPT
from configs.gaia.gpt_prompts import PLANNER_PROMPT as GPT_PLANNER_PROMPT
from configs.gaia.gpt_prompts import PLANNER_PROMPT_REPLAN as GPT_PLANNER_PROMPT_REPLAN

CONFIGS = {
    "default_model": "gpt-4.1",
    "prompts": {
        "gpt": {
            "planner_prompt": GPT_PLANNER_PROMPT,
            "planner_prompt_replan": GPT_PLANNER_PROMPT_REPLAN,
            "planner_critic_prompt": GPT_PLANNER_CRITIC_PROMPT,
            "output_prompt": GPT_OUTPUT_PROMPT,
            "output_prompt_final": GPT_OUTPUT_PROMPT_FINAL,
        },
        "llama": {
            "planner_prompt": GPT_PLANNER_PROMPT,
            "planner_prompt_replan": GPT_PLANNER_PROMPT_REPLAN,
            "planner_critic_prompt": GPT_PLANNER_CRITIC_PROMPT,
            "output_prompt": GPT_OUTPUT_PROMPT,
            "output_prompt_final": GPT_OUTPUT_PROMPT_FINAL,
        },
    },
    "max_replans": 3,
}
