from src.dynacall.constants import JOINNER_FINISH


_PARALLELQA_POLICY = """Return exactly one JSON array and nothing else.
Top-level format: [{"kind":"tool",...}, ...].
Allowed node kinds: tool, join.

Node formats:
- tool: {"kind":"tool","tool":"ToolName","args":[...]}
- tool with reusable id: {"kind":"tool","id":"name","tool":"ToolName","args":[...]}
- join: {"kind":"join"}

Hard constraints:
1. Output valid JSON only. No markdown fences, no comments, no trailing commas.
2. Use only tools: Search, semantic_map, Calculate.
3. Use named references only (for example "$everest_h", "$ratio").
4. End with exactly one final {"kind":"join"}.
5. semantic_map is numeric-extraction only in this benchmark. Do not use semantic_map for comparison, ranking, winner selection, or entity/name selection.

Planning policy:
1. Search for each required entity first.
2. Extract scalar numeric facts with semantic_map before any arithmetic.
3. Use Calculate for deterministic math only.
4. For comparison questions, perform all comparisons/ranking with Calculate steps; do not delegate comparison logic to semantic_map.
5. If the question asks for a numeric result, plan to produce a numeric value before join.
6. Before any Calculate step, each numeric dependency must be validated as a numeric literal (digits/decimal only).
7. If semantic_map for schema "number" returns unknown/non-numeric text, do not continue to Calculate; add a recovery step (re-search or stricter re-extraction) first.
8. Ratio rule: when comparing ratios, use consistent directional definition. By default use ratio(A,B)=max(A,B)/min(A,B) (ratio >= 1) unless the question explicitly specifies direction "A to B".
9. Reuse rule: one Search result can feed multiple semantic_map calls to extract different numeric fields; do not repeat Search when the needed fields are already in the same retrieved text.
"""


_PARALLELQA_FEWSHOTS = """Question: If Mount Everest shortens by 1%, Mount Kilimanjaro grows by 7%, Aconcagua decreases by 3%, and K2 extends by 2%, what is the ratio between the heights of the tallest and shortest mountains?
[
  {"kind":"tool","id":"everest_s","tool":"Search","args":["Mount Everest"]},
  {"kind":"tool","id":"everest_h","tool":"semantic_map","args":["Extract Mount Everest height in meters as one number.",["$everest_s"],"number"]},
  {"kind":"tool","id":"kili_s","tool":"Search","args":["Mount Kilimanjaro"]},
  {"kind":"tool","id":"kili_h","tool":"semantic_map","args":["Extract Mount Kilimanjaro height in meters as one number.",["$kili_s"],"number"]},
  {"kind":"tool","id":"acon_s","tool":"Search","args":["Aconcagua"]},
  {"kind":"tool","id":"acon_h","tool":"semantic_map","args":["Extract Aconcagua height in meters as one number.",["$acon_s"],"number"]},
  {"kind":"tool","id":"k2_s","tool":"Search","args":["K2"]},
  {"kind":"tool","id":"k2_h","tool":"semantic_map","args":["Extract K2 height in meters as one number.",["$k2_s"],"number"]},
  {"kind":"tool","id":"everest_adj","tool":"Calculate","args":["$everest_h*0.99",["$everest_h"]]},
  {"kind":"tool","id":"kili_adj","tool":"Calculate","args":["$kili_h*1.07",["$kili_h"]]},
  {"kind":"tool","id":"acon_adj","tool":"Calculate","args":["$acon_h*0.97",["$acon_h"]]},
  {"kind":"tool","id":"k2_adj","tool":"Calculate","args":["$k2_h*1.02",["$k2_h"]]},
  {"kind":"tool","id":"ratio","tool":"Calculate","args":["max($everest_adj,$kili_adj,$acon_adj,$k2_adj)/min($everest_adj,$kili_adj,$acon_adj,$k2_adj)",["$everest_adj","$kili_adj","$acon_adj","$k2_adj"]]},
  {"kind":"join"}
]
###

Question: If the Pacific Ocean shrank by 1.5 times and the Indian Ocean expands by 2.5 times, which ocean will have larger area?
[
  {"kind":"tool","id":"pacific_s","tool":"Search","args":["Pacific Ocean"]},
  {"kind":"tool","id":"pacific_area","tool":"semantic_map","args":["Extract Pacific Ocean area in square kilometers as one number.",["$pacific_s"],"number"]},
  {"kind":"tool","id":"indian_s","tool":"Search","args":["Indian Ocean"]},
  {"kind":"tool","id":"indian_area","tool":"semantic_map","args":["Extract Indian Ocean area in square kilometers as one number.",["$indian_s"],"number"]},
  {"kind":"tool","id":"pacific_adj","tool":"Calculate","args":["$pacific_area/1.5",["$pacific_area"]]},
  {"kind":"tool","id":"indian_adj","tool":"Calculate","args":["$indian_area*2.5",["$indian_area"]]},
  {"kind":"tool","id":"larger_val","tool":"Calculate","args":["max($pacific_adj,$indian_adj)",["$pacific_adj","$indian_adj"]]},
  {"kind":"join"}
]
###

Question: If Venus radius increases by 10% and Mars radius decreases by 20%, what is the difference between their new radii?
[
  {"kind":"tool","id":"venus_s","tool":"Search","args":["Venus"]},
  {"kind":"tool","id":"venus_r","tool":"semantic_map","args":["Extract Venus mean radius in kilometers as one number.",["$venus_s"],"number"]},
  {"kind":"tool","id":"mars_s","tool":"Search","args":["Mars"]},
  {"kind":"tool","id":"mars_r","tool":"semantic_map","args":["Extract Mars mean radius in kilometers as one number.",["$mars_s"],"number"]},
  {"kind":"tool","id":"venus_new","tool":"Calculate","args":["$venus_r*1.10",["$venus_r"]]},
  {"kind":"tool","id":"mars_new","tool":"Calculate","args":["$mars_r*0.80",["$mars_r"]]},
  {"kind":"tool","id":"diff","tool":"Calculate","args":["abs($venus_new-$mars_new)",["$venus_new","$mars_new"]]},
  {"kind":"join"}
]
###
"""


PLANNER_PROMPT = _PARALLELQA_POLICY + "\n\n" + _PARALLELQA_FEWSHOTS
PLANNER_PROMPT_REPLAN = _PARALLELQA_POLICY + "\n\n" + _PARALLELQA_FEWSHOTS


OUTPUT_PROMPT = (
    "Solve a question answering task with interleaving Observation, Thought, and Action steps. "
    "Answer should always be a single item and MUST not be multiple choices.\n"
    "You must output exactly two lines in this order and nothing else:\n"
    "Line 1: Thought: <very short reasoning>\n"
    f"Line 2: Action: {JOINNER_FINISH}(<final_answer>)\n"
    "Do not put Action or Finish on the Thought line. Do not merge the two lines.\n"
    "Do not output markdown/code fences, bullets, or extra text before/after these two lines.\n"
    "Action can be only one type:"
    f" (1) {JOINNER_FINISH}(answer): returns the answer and finishes the task. "
    "    - Final answer MUST NOT contain any description, and must be short (e.g. Yes/No, numbers, entity names, etc.)\n"
    "    - When you are asked for differences, you consider the absolute value of the difference.\n"
    "    - STRICT VALUE RULE: if the question asks for a value (ratio/difference/average/sum/product/percentage/rate), "
    "the answer MUST be a numeric literal only.\n"
    "    - For value questions, output digits (and decimal point if needed) only. No unit, no words, no labels, no explanation.\n"
    "    - Forbidden for value questions: entity names, equations, comparisons, units, or explanatory phrases.\n"
    "    - Required format for value questions: return only the numeric value in Finish(...).\n"
    "    - If unsure between an entity and a value, prefer the value that matches the question target type.\n"
    "    - Never output 'Thought: ...Finish(...)'. This is invalid.\n"
    "    - TYPE LOCK (value question): Finish(...) must contain only a numeric literal; never output entity names, pair names, or comparison phrases.\n"
    "    - TYPE LOCK (entity question): Finish(...) must contain exactly one entity name; never output pair/comparison phrases such as 'A vs B', 'A and B', or 'larger/smaller is ...'.\n"
    "    - TARGET-TYPE RULE: if the question asks which value/ratio/difference/number is larger/smaller, answer with the value only; "
    "if the question asks which entity/object/place/person is larger/smaller/faster/deeper, answer with the entity name only.\n"
)
