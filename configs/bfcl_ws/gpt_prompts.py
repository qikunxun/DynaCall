from src.dynacall.constants import JOINNER_FINISH, JOINNER_REPLAN

_BFCL_POLICY = """Return exactly one JSON array and nothing else.
Top-level format: [{"kind":"tool",...}, ...].
Allowed node kinds: tool, branch, replan, join.

Node formats:
- tool: {"kind":"tool","tool":"ToolName","args":[...]}
- tool with reusable id: {"kind":"tool","id":"name","tool":"ToolName","args":[...]}
- branch: {"kind":"branch","condition":"predicate or structured condition","then":[...],"else":[...]}
- replan: {"kind":"replan","scope":"local"|"global","reason":"short reason"}
- join: {"kind":"join"}

Core rules:
1. Output valid JSON only.
2. Non-terminal plans must not include join.
3. Use named ids only (e.g., $s1, $m1). Never use positional refs like $1/$2 or dotted refs like $m1.country.
4. semantic_map outputs must be single scalar entities/values; downstream steps must reference whole ids (e.g., $m1).
5. If multiple entities are needed, you MUST use multiple semantic_map calls (one entity per call).
5a. You MUST NOT use one semantic_map call to jump directly to the final answer when intermediate entities are required by the question.
6. Preferred loop: search_engine -> semantic_map -> branch.
7. Branch is a hard gate: empty/wrong-type/ungrounded extraction MUST go to else (fallback or replan), never then.
8. Multi-hop plans must preserve entity-chain consistency: each hop must be explicitly linked to the previous hop entity.
8a. For multi-hop questions, plan shape MUST be: search -> semantic_map(hop1) -> branch -> search/fetch -> semantic_map(hop2) -> ... -> semantic_map(final target) -> branch -> join.
9. Enforce target type strictly (person/name/year/number/title). Never return intermediate entities.
10. For time-sensitive targets (e.g., "as of April 2025"), keep exact temporal anchor in query and evidence.
11. For ambiguous entities (same-name people/orgs/places), add a disambiguation probe before finalizing.
12. For "born at / headquartered at / worked at" relations, confirm exact relation wording before downstream extraction.
13. Use authoritative/official sources first for critical slots; treat low-trust mirrors/listicles as weak evidence.
14. Local replan should change the weak hop/query (tighter entity + date + role anchors), not repeat the same route.
15. Never pass placeholders (unknown/N/A/empty) to downstream tools.
16. Keep output_schema scalar plain-text only (e.g., person:string, year:string, beds:string); no combined schemas.
16a. Do not ask semantic_map to output "final answer" before required intermediate hops are extracted and validated.
"""

_BFCL_FEW_SHOTS = """Question: Multi-hop person target with possible empty middle slot; enforce local replan.
[
  {"kind":"tool","id":"s1","tool":"search_engine","args":["2024 literature award winner alma mater current university president April 2025 official"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract the 2024 literature award winner person name only.",["$s1"],"winner:string"]},
  {
    "kind":"branch",
    "condition":"is_nonempty($m1)",
    "then":[
      {"kind":"tool","id":"s2","tool":"search_engine","args":["$m1 alma mater university official biography"]},
      {"kind":"tool","id":"m2","tool":"semantic_map","args":["Extract the alma mater university name only.",["$s2"],"university:string"]},
      {
        "kind":"branch",
        "condition":"is_nonempty($m2)",
        "then":[
          {"kind":"tool","id":"s3","tool":"search_engine","args":["$m2 current president April 2025 official"]},
          {"kind":"tool","id":"m3","tool":"semantic_map","args":["Extract the current president person name as of April 2025.",["$s3"],"person:string"]},
          {
            "kind":"branch",
            "condition":"is_nonempty($m3)",
            "then":[{"kind":"join"}],
            "else":[{"kind":"replan","scope":"local","reason":"Final president hop unresolved."}]
          }
        ],
        "else":[{"kind":"replan","scope":"local","reason":"Alma mater hop unresolved."}]
      }
    ],
    "else":[
      {"kind":"tool","id":"p1","tool":"fetch_urls","args":[["$s1"]]},
      {"kind":"tool","id":"m4","tool":"semantic_map","args":["Extract the winner person name only from fetched pages.",["$p1"],"winner:string"]},
      {"kind":"branch","condition":"is_nonempty($m4)","then":[{"kind":"join"}],"else":[{"kind":"replan","scope":"local","reason":"Winner hop unresolved after fetch fallback."}]}
    ]
  }
]
###

Question: Multi-hop numeric target; if scalar missing/non-digit, local replan instead of forcing answer.
[
  {"kind":"tool","id":"s1","tool":"search_engine","args":["target hospital official profile licensed beds"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract hospital name only.",["$s1"],"hospital:string"]},
  {
    "kind":"branch",
    "condition":"is_nonempty($m1)",
    "then":[
      {"kind":"tool","id":"s2","tool":"search_engine","args":["$m1 licensed bed count official profile"]},
      {"kind":"tool","id":"m2","tool":"semantic_map","args":["Extract final bed_count digits only.",["$s2"],"bed_count:string"]},
      {"kind":"branch","condition":"is_nonempty($m2)","then":[{"kind":"join"}],"else":[{"kind":"replan","scope":"local","reason":"Numeric final target still unsupported."}]}
    ],
    "else":[
      {"kind":"tool","id":"p1","tool":"fetch_urls","args":[["$s1"]]},
      {"kind":"tool","id":"m3","tool":"semantic_map","args":["Extract hospital name only from fetched pages.",["$p1"],"hospital:string"]},
      {"kind":"branch","condition":"is_nonempty($m3)","then":[{"kind":"join"}],"else":[{"kind":"replan","scope":"local","reason":"Hospital hop unresolved."}]}
    ]
  }
]
###

Question: Two-hop chain where hop-1 may be correct but hop-2 is weak; must local-replan hop-2.
[
  {"kind":"tool","id":"s1","tool":"search_engine","args":["target author birthplace state"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract birthplace state entity only.",["$s1"],"state:string"]},
  {
    "kind":"branch",
    "condition":"is_nonempty($m1)",
    "then":[
      {"kind":"tool","id":"s2","tool":"search_engine","args":["largest lake in $m1"]},
      {"kind":"tool","id":"m2","tool":"semantic_map","args":["Extract final target lake name only (not state).",["$s2"],"lake:string"]},
      {
        "kind":"branch",
        "condition":"is_nonempty($m2)",
        "then":[{"kind":"join"}],
        "else":[
          {"kind":"tool","id":"p2","tool":"fetch_urls","args":[["$s2"]]},
          {"kind":"tool","id":"m3","tool":"semantic_map","args":["Re-extract lake name from fetched geography pages.",["$p2"],"lake:string"]},
          {
            "kind":"branch",
            "condition":"is_nonempty($m3)",
            "then":[{"kind":"join"}],
            "else":[{"kind":"replan","scope":"local","reason":"Second hop unresolved after fetch fallback."}]
          }
        ]
      }
    ],
    "else":[{"kind":"replan","scope":"local","reason":"First hop unresolved."}]
  }
]
###"""

PLANNER_PROMPT = _BFCL_POLICY + "\nExamples:\n\n" + _BFCL_FEW_SHOTS

PLANNER_PROMPT_REPLAN = _BFCL_POLICY + """

Replanning rules:
1. Produce a fully self-contained new JSON array.
2. Do not reference ids from the failed plan.
3. Keep useful grounded observations for local replan.
4. Change the weak probe/query rather than repeating the same failed route.
5. If a concrete URL/object is already grounded, continue with read/extract instead of rediscovery.
6. Replan should usually be shorter and more decisive than the failed attempt.
7. Do not output join in intermediate replans.
8. If the failure was empty required slots, the new plan must add a stronger retrieval/extraction step for those exact slots.
"""

OUTPUT_PROMPT = (
    "Solve a QA task from existing observations. Return only one action.\n"
    f"Action can be: (1) {JOINNER_FINISH}(answer), or (2) {JOINNER_REPLAN}(missing_information).\n"
    "- Use Finish when a grounded observation already supports a concise final answer.\n"
    "- Use Replan when evidence is missing, contradicted, or still ambiguous.\n"
    "- Final answer must be a short single item (name, number, date, URL, or short phrase) with no extra commentary.\n"
    "- The final answer type must exactly match the question target (e.g., person name, lake name, year, number), not an intermediate hop.\n"
    "- If required multi-hop slots are missing/empty in observations, prefer Replan over emitting an intermediate entity.\n"
    "- Prefer the most specific grounded answer type requested by the question; do not substitute a broader entity.\n"
    "- Do not output tool calls in this stage.\n"
)
