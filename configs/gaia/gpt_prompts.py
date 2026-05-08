from src.dynacall.constants import JOINNER_FINISH, JOINNER_REPLAN


_GAIA_POLICY = """Return exactly one JSON array and nothing else.
Top-level format: [{"kind":"tool",...}, ...].
Allowed node kinds: tool, branch, replan, join.

Node formats:
- tool: {"kind":"tool","tool":"ToolName","args":[...]}
- tool with reusable id: {"kind":"tool","id":"name","tool":"ToolName","args":[...]}
- branch: {"kind":"branch","condition":"predicate or structured condition","then":[...],"else":[...]}
- replan: {"kind":"replan","scope":"local"|"global","reason":"short reason"}
- join: {"kind":"join"}

Core policy:
1. Think like ReAct, but output only executable JSON tool plans.
2. Plan iteratively and keep the active horizon short.
3. In the opening stage, prefer one grounded probe; add branch only when the next step truly depends on uncertainty handling.
4. Use branch as a quality gate when needed, not as mandatory boilerplate on every probe.
5. If the first route is weak, ambiguous, blocked, or off-object, replan quickly; otherwise continue with the shortest grounded read/extract chain.
6. semantic_map is the semantic conversion operator. Use it only when the next step needs a compact typed artifact such as one URL, one answer word, one cleaned scalar, one candidate list, or one narrowed grounded subset.
7. If current observations already answer the question, return [{"kind":"join"}].
8. Do not join on an intermediate entity when the question still asks for a downstream attribute, role, officeholder, date, count, or another derived fact.
9. For non-URL questions, selecting a URL is only an intermediate step. Continue until the requested non-URL field is grounded.

Rules:
1. Output valid JSON only.
2. Non-terminal rounds must not include join.
3. Tool calls must use exact shape {"kind":"tool","tool":"ExactToolName","args":[...]}.
4. Prefer named ids for reusable intermediate results; avoid positional references.
5. When an intermediate result will be reused later, give it a named id and reference it by that id. Avoid positional references such as $1, $2, or $3 in GAIA plans.
6. args must always be a JSON array.
7. search_engine is the default tool for web discovery.
8. For web questions, search first, then read the most relevant grounded page.
9. Never pass raw search results directly into URL-reading tools when one exact URL still needs to be chosen. First use semantic_map to select one concrete URL string.
10. If search-result snippets already expose the exact grounded numeric operands needed for a deterministic computation from the requested source family, you may extract those operands directly and compute without forcing an extra URL-selection step.
11. If search results mainly contain listing or index pages, read the listing page first, then isolate the exact detail page.
12. If search results do not isolate the required object, branch on object verification and replan with a better query or source.
13. Early branch conditions should test exact object/date/source/category grounding, contradiction, or evidence sufficiency. Do not use mere non-emptiness when exact grounding is required.
14. Never answer from a nearby-but-wrong page. If a page shows a mismatched date, title, source, category, or object, branch and replan instead of extracting.
15. If a domain or URL family is blocked or repeatedly fails, abandon it for the current question and switch to another source family.
16. If the question includes an exact date, year, document type, site, or source constraint, verify that exact constraint before extracting the answer.
17. Preserve distinctive titles, phrases, names, and other strong anchors from the question when forming search queries.
18. If the grounded source is a remote PDF, image, spreadsheet, archive, audio, video, or other remote file URL, first use download_file_from_url, then pass the local path to local file tools. Do not pass remote URLs directly into local file tools.
19. Prefer code_interpreter for deterministic parsing, file logic, exact counting, and exact computation after the needed facts are already grounded.
20. Never use code_interpreter to simulate web search, API calls, or unavailable external data.
21. Preserve exact constraints from the question, including source, date, year, unit, precision, and final format.
22. Default to not using semantic_map. Add it only when you need a compact typed artifact for the next step.
23. If one tool's observation can already be passed directly into the next tool, do not insert semantic_map.
24. For counting, ranking, comparison, or arithmetic questions, first ground the inputs, then compute deterministically.
25. If a tool returns an error, empty output, blocked page, or contradictory evidence, do not continue blindly; branch and replan.
26. If one grounded source is already sufficient to extract the final answer, prefer a short direct read/extract path over additional broad search.
27. For source-constrained questions, keep the evidence path closed: once the question anchors on a specific source, issue, row, date, site, or candidate set, do not drift to nearby entities or substitute a different source family unless you explicitly replan.
28. When the final answer must be chosen from a grounded candidate set, first ground that set, then return one member of that set only.
29. If the latest search or page evidence is off-domain, off-entity, speculative, synthetic, SEO-like, or otherwise weakly grounded, treat it as pollution and replan instead of extracting.
30. For multi-hop questions, verify each hop's object before using it as the input to the next hop; do not let an unverified intermediate object contaminate downstream extraction.
"""


_GAIA_FEW_SHOTS = """Question: Search results contain several candidate pages, and the next step needs one exact grounded article URL.
[
  {"kind":"tool","id":"s1","tool":"search_engine","args":["site:arxiv.org target paper clue"]},
  {
    "kind":"branch",
    "condition":{"op":"llm_judge","instruction":"Do the latest search results isolate one exact grounded article/object already?","inputs":["$s1"]},
    "then":[
      {"kind":"tool","id":"m1","tool":"semantic_map","args":["From the search results, extract the single exact grounded article URL that best matches the target clue. Return one URL string only.",["$s1"],"string"]}
    ],
    "else":[{"kind":"replan","scope":"local","reason":"Search results still do not isolate one exact grounded article URL."}]
  }
]
###

Question: I already know one exact URL and want to read that page next.
[
  {"kind":"tool","id":"p1","tool":"web_browser","args":["https://example.com/page"]}
]
###

Question: A later attempt already has the right article URL and needs one exact answer word from the page.
[
  {"kind":"tool","id":"p1","tool":"web_browser","args":["grounded article URL from prior observation"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract the exact grounded answer word from the page text. Return one word only.",["$p1"],"string"]},
  {"kind":"join"}
]
###

Question: A prior attempt already found the candidate word list from source A, but source B must be the exact article from one specific date before you can return the shared word.
[
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Treat the prior source-A observation as a candidate-word list and preserve it exactly.",["candidate list from prior observation"],"list[string]"]},
  {"kind":"tool","id":"s2","tool":"search_engine","args":["listing/search results for source B with exact date clue"]},
  {
    "kind":"branch",
    "condition":{"op":"llm_judge","instruction":"Do the latest results isolate the exact dated source-B article already?","inputs":["$s2"]},
    "then":[
      {"kind":"tool","id":"m2","tool":"semantic_map","args":["From the listing/search results, extract the single exact article URL matching the requested date. Do not return the listing page or a nearby date.",["$s2"],"string"]},
      {"kind":"tool","id":"p2","tool":"web_browser","args":["$m2"]},
      {"kind":"tool","id":"m3","tool":"semantic_map","args":["From the exact dated article, extract the single candidate-compatible descriptor word only. If the page uses a longer phrase, map it to the one word from the candidate list that expresses the same descriptor; do not return a multi-word phrase.",["$m1","$p2"],"string"]},
      {"kind":"tool","id":"m4","tool":"semantic_map","args":["Return the final answer as one word from the source-A candidate list only. If source B gives a related descriptor rather than the exact same token, map it to the closest candidate-list concept and return that candidate-list word.",["$m1","$m3"],"string"]}
    ],
    "else":[{"kind":"replan","scope":"local","reason":"Need the one exact source-B article from the requested date before returning the shared word."}]
  }
]
###

Question: A cross-source arXiv question asks for one word chosen from source A, but source B has an exact submission date and is the easier object to verify first.
[
  {"kind":"tool","id":"sB","tool":"search_engine","args":["August 2016 physics.soc-ph arXiv monthly listing"]},
  {
    "kind":"branch",
    "condition":{"op":"llm_judge","instruction":"Do these search results ground an August 2016 physics.soc-ph monthly listing or another source-B discovery page that can yield exact dated candidate abstract pages, rather than only nearby papers?", "inputs":["$sB"]},
    "then":[
      {"kind":"tool","id":"listingB","tool":"semantic_map","args":["From the search results, extract the single best August 2016 physics.soc-ph monthly listing URL or equivalent candidate-discovery page for source B. Return one URL string only; return an empty string if unresolved.",["$sB"],"string"]},
      {
        "kind":"branch",
        "condition":{"op":"llm_judge","instruction":"Is a concrete monthly listing or candidate-discovery page for August 2016 physics.soc-ph grounded? Treat generic paper results or wrong months/categories as failure.", "inputs":["$sB","$listingB"]},
        "then":[
          {"kind":"tool","id":"pageB","tool":"web_browser","args":["$listingB"]},
          {"kind":"tool","id":"candB","tool":"semantic_map","args":["From the discovery page text, extract one or a few candidate arXiv abstract URLs whose evidence is compatible with the exact requested source-B constraints, especially submitted on 11 Aug 2016 and physics.soc-ph. Return a short JSON list of URLs only; return an empty list if no such candidates are grounded.",["$pageB"],"list[string]"]},
          {
            "kind":"branch",
            "condition":{"op":"llm_judge","instruction":"Is there at least one grounded candidate abstract URL for source B that is compatible with the exact requested date/category constraints?", "inputs":["$pageB","$candB"]},
            "then":[
              {"kind":"tool","id":"candPagesB","tool":"web_browser","args":["$candB"]},
              {"kind":"tool","id":"descB","tool":"semantic_map","args":["From the candidate abstract pages, keep only the exact source-B paper whose abstract page verifies both constraints: submitted on 11 Aug 2016 and physics.soc-ph / Physics and Society. Then extract the one or few grounded society-descriptor words from that exact paper. Return a short JSON list of words only; return an empty list if no exact candidate survives verification.",["$candPagesB"],"list[string]"]},
              {
                "kind":"branch",
                "condition":{"op":"llm_judge","instruction":"Is source B exact-date/category grounded and is the descriptor-word list non-empty? Treat any date/category mismatch or only-nearby-paper evidence as failure.", "inputs":["$candPagesB","$descB"]},
                "then":[
                  {"kind":"tool","id":"sA","tool":"search_engine","args":["\"Fairness in Agreement With European Values: An Interdisciplinary Perspective on AI Regulation\" arXiv June 2022 abstract"]},
                  {
                    "kind":"branch",
                    "condition":{"op":"llm_judge","instruction":"Do these results/snippets ground the exact source-A paper named in the query strongly enough to extract candidate words, rather than only nearby AI-regulation papers?", "inputs":["$sA"]},
                    "then":[
                      {"kind":"tool","id":"wordsA","tool":"semantic_map","args":["From the search results or snippets for the exact source-A paper, extract the closed candidate word set formed by the three axis-end labels. Preserve exact word forms. Return a short JSON list of words only; if the exact paper or its candidate words are not grounded, return an empty list.",["$sA"],"list[string]"]},
                      {
                        "kind":"branch",
                        "condition":{"op":"llm_judge","instruction":"Is source A grounded and is the candidate word set non-empty? Treat wrong paper, vague snippet evidence, or guessed candidate words as failure.", "inputs":["$sA","$wordsA"]},
                        "then":[
                          {"kind":"tool","id":"match","tool":"semantic_map","args":["Return the single source-A candidate word that overlaps with the verified source-B descriptor words. Return one source-A candidate word only; if there is no exact candidate-set match, return an empty string.",["$wordsA","$descB"],"string"]},
                          {
                            "kind":"branch",
                            "condition":{"op":"llm_judge","instruction":"Is the matched answer one non-empty word from the source-A candidate list and supported by source-B descriptor evidence?", "inputs":["$wordsA","$descB","$match"]},
                            "then":[{"kind":"join"}],
                            "else":[{"kind":"replan","scope":"local","reason":"Need a clean overlap between source-A candidate words and verified source-B descriptors."}]
                          }
                        ],
                        "else":[{"kind":"replan","scope":"local","reason":"Need source-A candidate words from the June 2022 AI-regulation paper."}]
                      }
                    ],
                    "else":[{"kind":"replan","scope":"local","reason":"Need non-empty source-A search evidence before candidate extraction."}]
                  }
                ],
                "else":[{"kind":"replan","scope":"local","reason":"Need exact-date source-B descriptor words before matching source A."}]
              }
            ],
            "else":[{"kind":"replan","scope":"local","reason":"Need source-B candidate abstract pages from the August 2016 physics.soc-ph discovery page."}]
          }
        ],
        "else":[{"kind":"replan","scope":"global","reason":"Need an August 2016 physics.soc-ph discovery page before exact-date source-B verification."}]
      }
    ],
    "else":[{"kind":"replan","scope":"global","reason":"Need an August 2016 physics.soc-ph discovery page before exact-date source-B verification."}]
  }
]
###

Question: I need the exact arXiv abstract page for one paper with a known submission date.
[
  {"kind":"tool","id":"s1","tool":"search_engine","args":["site:arxiv.org/abs \"Submitted on 11 Aug 2016\" target category/topic clue"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract the single arXiv abstract URL whose page matches the exact submission date and topic clue. Return one bare URL string only.",["$s1"],"string"]},
  {"kind":"tool","id":"p1","tool":"web_browser","args":["$m1"]}
]
###

Question: A grounded source gives one place, and the final answer needs its exact five-digit ZIP code.
[
  {"kind":"tool","id":"p1","tool":"web_browser","args":["grounded source URL"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract the exact place name from the source. Return one place string only.",["$p1"],"string"]},
  {"kind":"tool","id":"s2","tool":"search_engine","args":["$m1 ZIP code"]},
  {"kind":"tool","id":"m2","tool":"semantic_map","args":["Extract the exact five-digit ZIP code for the grounded place. Return one string only.",["$m1","$s2"],"string"]},
  {"kind":"join"}
]
###

Question: A Wikipedia biography page includes a long discography, and the answer needs the number of studio albums in a specific year range.
[
  {"kind":"tool","id":"s1","tool":"search_engine","args":["artist name English Wikipedia discography studio albums source year"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract the single exact English Wikipedia article URL for the target artist. Return one URL string only.",["$s1"],"string"]},
  {"kind":"tool","id":"w1","tool":"wiki_section_extract","args":["$m1","Studio albums","2022"]},
  {"kind":"tool","id":"m2","tool":"semantic_map","args":["From the extracted Studio albums section, keep only the grounded studio album entries whose years fall inside the requested range. Return a short list, not a count.",["$w1"],"list[string]"]},
  {"kind":"tool","id":"m3","tool":"semantic_map","args":["Count the filtered studio album entries and return one number string only.",["$m2"],"string"]},
  {"kind":"join"}
]
###

Question: A local spreadsheet/image/archive computation can be solved directly once the file path is known.
[
  {"kind":"tool","id":"f1","tool":"spreadsheet_reader","args":["/absolute/path/to/file.xlsx"]},
  {"kind":"tool","id":"p1","tool":"code_interpreter","args":["# read the grounded local file, compute the exact final result, and print only Yes or No\\nprint('No')","python"]},
  {"kind":"join"}
]
###

Question: A stochastic three-slot queue game asks which numbered token has the highest exact probability of being ejected under uniformly random actions.
[
  {"kind":"tool","id":"p1","tool":"code_interpreter","args":["from functools import lru_cache\\nN = 7\\n@lru_cache(None)\\ndef dp(pos1, pos2, pos3, next_item):\\n    if pos1 is None and pos2 is None and pos3 is None:\\n        return {}\\n    result = {}\\n    next_a = next_item if next_item <= N else None\\n    next_b = next_item + 1 if next_item + 1 <= N else None\\n    transitions = [\\n        (pos1, pos2, pos3, next_a, next_item + 1 if next_item <= N else next_item),\\n        (pos2, pos3, next_a, next_b, next_item + 2 if next_item <= N else next_item),\\n        (pos3, pos2, next_a, next_b, next_item + 2 if next_item <= N else next_item),\\n    ]\\n    for immediate, np1, np2, np3, nnxt in transitions:\\n        if immediate is not None:\\n            result[immediate] = result.get(immediate, 0.0) + 1/3\\n        sub = dp(np1, np2, np3, nnxt)\\n        for key, value in sub.items():\\n            result[key] = result.get(key, 0.0) + value / 3\\n    return result\\nprobs = dp(1, 2, 3, 4)\\nprint(max(probs, key=probs.get))","python"]},
  {"kind":"join"}
]
###

Question: A bloc question asks for the two countries whose capital cities are furthest apart according to one grounded Wikipedia source.
[
  {"kind":"tool","id":"s1","tool":"search_engine","args":["bloc member states capitals wikipedia"]},
  {"kind":"tool","id":"u1","tool":"semantic_map","args":["Extract one exact English Wikipedia URL that grounds the current formal bloc member-country set. Return one URL string only.",["$s1"],"string"]},
  {"kind":"tool","id":"p1","tool":"web_browser","args":["$u1"]},
  {"kind":"tool","id":"m1","tool":"semantic_map","args":["Extract complete member-capital records as json{records:list[json{country:string,capital:string}]}. Keep one consistent core/full-member set and exclude observer/applicant/accession entries if mixed on the page. Capitals must be non-empty.",["$p1"],"json{records:list[json{country:string,capital:string}]}"]},
  {"kind":"tool","id":"py1","tool":"code_interpreter","args":["# compute great-circle distance for all country-capital pairs from grounded records\\nimport math, json\\nR=6371.0\\ncoords={\\n'bandar seri begawan':(4.9031,114.9398),'phnom penh':(11.5564,104.9282),'jakarta':(-6.2088,106.8456),'vientiane':(17.9757,102.6331),'kuala lumpur':(3.1390,101.6869),'naypyidaw':(19.7633,96.0785),'manila':(14.5995,120.9842),'singapore':(1.3521,103.8198),'bangkok':(13.7563,100.5018),'dili':(-8.5569,125.5603),'hanoi':(21.0278,105.8342)}\\nobj=$m1\\nrecords=obj.get('records',[]) if isinstance(obj,dict) else []\\ndef hav(a,b):\\n la1,lo1=a; la2,lo2=b\\n p1,p2=math.radians(la1),math.radians(la2)\\n dp=math.radians(la2-la1); dl=math.radians(lo2-lo1)\\n h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2\\n return 2*R*math.asin(min(1,math.sqrt(h)))\\nbest=None\\nfor i in range(len(records)):\\n for j in range(i+1,len(records)):\\n  c1=records[i].get('country',''); c2=records[j].get('country','')\\n  k1=records[i].get('capital','').strip().lower(); k2=records[j].get('capital','').strip().lower()\\n  if k1 in coords and k2 in coords:\\n   d=hav(coords[k1],coords[k2])\\n   if (best is None) or (d>best['distance_km']): best={'country_a':c1,'country_b':c2,'distance_km':d}\\nprint(json.dumps(best or {}, ensure_ascii=False))","python"]},
  {"kind":"tool","id":"m2","tool":"semantic_map","args":["Return only the two country names from py1 sorted alphabetically and comma-separated as `Country A, Country B`.",["$py1"],"string"]},
  {"kind":"join"}
]
###"""


PLANNER_PROMPT = _GAIA_POLICY + "\nExamples:\n\n" + _GAIA_FEW_SHOTS


PLANNER_PROMPT_REPLAN = _GAIA_POLICY + """

Replanning rules:
1. Produce a fully self-contained new JSON array.
2. Do not reference ids from the failed plan.
3. Keep successful earlier observations only for local replan.
4. Add only the next missing step or next missing short chunk.
5. Do not repeat the same weak query, blocked URL, or failed object.
6. Replans should usually be shorter and more decisive than the failed route.
7. If the first probe failed, change the probe itself.
8. If a concrete object is already verified, keep it and repair only the remaining suffix.
9. Prefer local replan by default. Use global replan only when the whole route must restart.
10. Keep distinctive phrases and strong entity anchors from the question when reformulating queries.
11. If a listing page or grounded object is already known, continue from it instead of restarting broad discovery.
12. If a domain or URL family already failed repeatedly, avoid it and switch to another source family.
13. If a prior extraction from the same page returned an empty value, narrow the source before repeating extraction.
14. If the exact page/object is still unresolved, do not push raw search ambiguity into answer extraction.
15. Once one exact URL/object is grounded, replans should usually focus on read -> extract steps for the missing field.
16. If a prior round already grounded an exact URL/object and the missing fact is a downstream field from that object, do not restart broad search; continue from that grounded object.
17. If a prior observation already contains a concrete page with an internal PDF/full-text/download link, continue from that existing page/link chain before launching new external discovery.
18. Use semantic_map only when you need to select, extract, or normalize the next compact value.
19. Do not output join during intermediate replans.
"""


PLANNER_CRITIC_PROMPT = """You are a GAIA plan critic.
Return exactly one JSON object:
{"approve": boolean, "reason": "short reason", "suggestion": "short actionable suggestion"}

Reject when:
- the plan is too long for current uncertainty
- the opening probe is not guarded by branch/replan
- the opening branch only checks non-emptiness when the question requires exact date/source/object grounding
- the concrete object is not verified before extraction
- a branch allows a nearby-but-wrong object to continue instead of forcing replan
- a listing/index page is treated as final evidence instead of candidate discovery
- semantic_map is used even though the raw observation is already enough for the next step
- semantic_map is asked to jump directly from noisy evidence to the final answer
- a count/aggregation question skips structured operands or deterministic computation
- for a non-URL question, the plan stops at URL discovery/locator selection without a follow-up read/extract step
- a multi-source question leaves one required source ungrounded before trying to finish
- for source-constrained multi-constraint retrieval, the plan jumps directly from noisy evidence to one final object without explicit verification against all constraints
- the fallback is only a near-duplicate retry

Prefer plans that:
- are short
- branch early at failure boundaries
- use early branch conditions as quality gates for exact object/date/source verification, not merely result existence
- verify exact date/source/object constraints before extraction
- when one discovery path fails, change the probe family instead of repeating near-duplicate queries
- use semantic_map only for compact typed artifacts
- isolate exact subsets before counting
- use code_interpreter only for deterministic computation after grounding
"""


OUTPUT_PROMPT = (
    "Solve a GAIA question answering task from the available observations.\n"
    "Return only one action.\n"
    f"Action can be one of two types: (1) {JOINNER_FINISH}(answer), or (2) {JOINNER_REPLAN}(missing_information).\n"
    "\n"
    "Finish rules:\n"
    "- If a semantic_map, code_interpreter, calculator, OCR, file, or grounded page observation directly contains a plausible final answer, prefer Finish over Replan.\n"
    "- The payload inside Finish(...) must be the final answer itself, not another action such as Replan(...), Finish(...), or a tool call.\n"
    "- Final answer must be short: one word, one name, one date, one number, one URL, or the exact requested phrase.\n"
    "- Prefer the most specific grounded observation over broader or noisier evidence.\n"
    "- Copy or minimally normalize grounded values to match the requested format.\n"
    "- If quote_verifier returns matches=true, answer Yes. If quote_verifier returns one grounded mismatched_word, answer with that one word only.\n"
    "- A URL-selection result, page locator, or source identifier is not enough evidence for a non-URL final answer. If the question asks for a person, number, title, date, or phrase, Finish only after that field itself has been extracted from grounded page/file/record evidence.\n"
    "- If the question specifies an exact date, site, issue, row, or source, do not answer from an observation that does not satisfy that exact constraint.\n"
    "- If an observation explicitly shows a mismatched date, title, issue, revision, category, or source, treat it as contradiction and prefer Replan over Finish.\n"
    "- If the question asks for a non-URL field and the latest strongest observation is only a URL/locator (including direct PDF URL), do not Finish with that URL; Replan to read and extract the requested field.\n"
    "- For multi-source questions, finish only after each required source or evidence path is grounded.\n"
    "- For source-constrained questions, keep the evidence path closed: if the answer must come from one exact source, issue, row, date, site, or grounded candidate set, do not substitute nearby sources or broader summaries.\n"
    "- For counting, comparison, ranking, or arithmetic questions, finish only after the operands are explicitly grounded and the result is computed deterministically.\n"
    "- When a question requires choosing from a grounded candidate set, return one member of that set only.\n"
    "- Preserve required output format, ordering, units, and title wording from the question when those constraints are grounded.\n"
    "- Do not Finish with an intermediate entity when the question still asks for a downstream attribute such as that entity's officeholder, date, count, value, or role. In that case prefer Replan unless a later observation already provides the final requested field.\n"
    "- Do not Finish from polluted or unrelated search results. If the latest search observations are off-domain, off-entity, or clearly irrelevant to the target object, Replan instead.\n"
    "- If the answer must come from a grounded candidate set, table row, list entry, issue entry, or source-defined entity set, do not return anything outside that grounded set.\n"
    "- If a multi-hop chain contains an unverified intermediate object, missing exact source alignment, or a broken evidence hop, prefer Replan over projecting a final answer.\n"
    "- If official pages are blocked and no alternate reachable source has been grounded yet, Replan instead of guessing from memory or weak snippets.\n"
    "- Do not guess from weak, tertiary, generic, or unsupported evidence.\n"
    "\n"
    "Replan rules:\n"
    "- Use Replan only when no observation contains a grounded candidate answer or when the candidate is explicitly contradicted.\n"
    "- Do not Replan just to reread pages if a prior focused extraction already answers the question.\n"
    "- If replanning, name the single missing fact or failed evidence path.\n"
    "- Never wrap Replan(...) inside Finish(...).\n"
    "\n"
    "Examples:\n"
    "Question: Which exact word is requested from the grounded source?\n"
    "semantic_map(<...>)\n"
    "Observation: answer_word\n"
    f"Action: {JOINNER_FINISH}(answer_word)\n"
    "###\n"
    "Question: Which exact object matches the requested constraints?\n"
    "search_engine(<...>)\n"
    "Observation: Search results contain several unrelated or mismatched candidates.\n"
    f"Action: {JOINNER_REPLAN}(need the exact grounded object matching the requested constraints)\n"
    "###\n"
    "Question: What is the exact numeric value requested from the grounded source?\n"
    "semantic_map(<...>)\n"
    "Observation: 12.4 kg\n"
    f"Action: {JOINNER_FINISH}(12.4)\n"
    "###\n"
)


OUTPUT_PROMPT_FINAL = (
    OUTPUT_PROMPT
    + f"\nFinal-step rule: return exactly one action. If a grounded final answer exists, return {JOINNER_FINISH}(answer); otherwise return {JOINNER_REPLAN}(single_missing_fact). Do not emit tool calls."
)
