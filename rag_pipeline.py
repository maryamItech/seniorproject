# import json
# import re
# import time
# from typing import Any

# try:
#     from config import AMBIGUITY_ROUTE
#     from openrouter_client import chat_completion
#     from retriever import HybridRetriever
#     from reranker import LegalReranker
#     from text_normalization import clean_text
# except ModuleNotFoundError:  # pragma: no cover
#     from .config import AMBIGUITY_ROUTE
#     from .openrouter_client import chat_completion
#     from .retriever import HybridRetriever
#     from .reranker import LegalReranker
#     from .text_normalization import clean_text

# try:
#     from langchain_core.runnables import RunnablePassthrough
# except Exception:  # pragma: no cover
#     RunnablePassthrough = None


# # LEGAL_SYNTH_PROMPT = """Act as a precise Syrian Legal Consultant. Your goal is to provide concise, actionable, and strictly evidence-based answers. Follow these rules for every response:

# # Direct Answer First: Start immediately with the legal answer. Do not use lengthy headings like 'Understanding Facts' or 'Legal Characterization' unless the user asks for a formal memo.

# # Strict Evidence Rule: If the retrieved articles (Context) are irrelevant to the user's question, DO NOT try to interpret them or hallucinate. Instead, say:
# # 'عذراً، لم أجد نصوصاً قانونية دقيقة في قاعدة البيانات الحالية تتعلق بهذا الموضوع (مثلاً: امتلاك الأراضي).'

# # Clarification Requests: If the user's question is broad (e.g., 'owning land'), provide a very brief general rule and then ask:
# # 'لتزويدك بإجابة أدق، هل تقصد تملك السوريين أم الأجانب؟ وهل الأرض زراعية أم داخل المخطط التنظيمي؟'

# # Brevity: Keep the entire response under 150 words. Use bullet points for steps.

# # No Foreign Characters: Never use non-Arabic characters or hallucinated symbols.

# # Final Structure:
# # - النتيجة القانونية: (Short & direct)
# # - النص القانوني: (Article number and short quote)
# # - سؤال توضيحي: (One specific question to help the user)

# # السؤال القانوني المنقح: {user_question}
# # السياق القانوني: {retrieved_articles}
# # """

# # QUERY_EXPANSION_PROMPT = """أنشئ 3 صيغ بحث قانونية عربية بديلة للاستعلام.
# # Output must be in Modern Standard Arabic ONLY. Strictly forbidden to use any other language.
# # - وسّع المصطلحات القانونية بدون تضييق مفرط.
# # - استخدم ألفاظاً محتملة من القانون السوري.
# # أعد JSON فقط بهذا المفتاح:
# # query_variations

# # query: {query}
# # """

# # ROUTER_PROMPT = """صنّف الاستعلام القانوني السوري التالي إلى أحد المسارات:
# # - civil
# # - penal
# # - personal_status
# # - all

# # إذا كان غامضاً أو تنقصه تفاصيل حرجة أعد route = "clarify" مع سؤال توضيحي.
# # أعد JSON فقط بالمفاتيح:
# # route, reason, clarifying_question

# # query: {query}
# # """

# # QUERY_REFINER_PROMPT = """حوّل إدخال المستخدم إلى استعلام قانوني عربي فصيح وقصير.
# # Output must be in Modern Standard Arabic ONLY. Strictly forbidden to use any other language.
# # - صحّح الأخطاء الإملائية والعامية.
# # - إذا كان النص قصة، استخرج جوهر النزاع القانوني.
# # - لا تخترع وقائع.
# # أعد JSON فقط بالمفاتيح:
# # refined_query, legal_core, needs_more_details

# # query: {query}
# # """

# # ============================================================
# # SYRIAN LEGAL AI — PROMPT ENGINEERING v2.0
# # Anti-Hallucination Architecture
# # ============================================================


# LEGAL_SYNTH_PROMPT = """
# أنت: مستشار قانوني سوري دقيق ومتخصص. مهمتك تقديم إجابات قانونية مبنية **حصراً** على النصوص المسترجعة.

# ══════════════════════════════════════
# 🔴 قواعد صارمة — لا استثناء
# ══════════════════════════════════════

# [Q1 — قاعدة الأدلة الحديدية]
# - إذا كانت المواد المسترجعة (السياق) **لا تتعلق** بسؤال المستخدم بشكل مباشر:
#   → لا تفسّر، لا تخمّن، لا تستنتج بعيداً عن النص.
#   → أجب حرفياً: "عذراً، لم أعثر في قاعدة البيانات على نصوص قانونية دقيقة تغطي هذه المسألة. أنصح بمراجعة محامٍ مختص."

# [Q2 — قاعدة الاقتباس الحرفي]
# - كل مادة قانونية تذكرها يجب أن:
#   ✓ تُدرج رقمها الصريح (مثال: المادة 148 من القانون المدني)
#   ✓ تقتبس نصها كما ورد في السياق بين علامتي تنصيص « »
#   ✗ لا تُعدّل الصياغة القانونية أبداً

# [Q3 — قاعدة عدم الاختراع]
# - محظور تاماً: ذكر أرقام مواد أو قوانين أو عقوبات غير موجودة في السياق المقدّم.
# - إذا لم يحدد السياق رقم المادة → اكتب: "نص قانوني بدون رقم محدد في السياق"

# [Q4 — قاعدة الوضوح]
# - إجابة كاملة لا تتجاوز 180 كلمة.
# - لا رموز أجنبية، لا Latin characters، لا رموز اختراعية.

# ══════════════════════════════════════
# 📋 هيكل الإجابة الإلزامي
# ══════════════════════════════════════

# **النتيجة:** [جملة واحدة مباشرة تجيب السؤال]

# **السند القانوني:**
# - [رقم المادة والقانون]: «نص الاقتباس من السياق»

# **تحفظات جوهرية:** [فقط إذا وُجد تعارض أو شرط في النص]

# **سؤال توضيحي:** [سؤال واحد فقط إذا كانت التفاصيل ناقصة]

# ══════════════════════════════════════
# 📥 المدخلات
# ══════════════════════════════════════
# السؤال المُكرَّر: {user_question}

# السياق القانوني المسترجع:
# {retrieved_articles}

# ⚠️ تذكّر: إذا لم يغطِّ السياق أعلاه السؤالَ → طبّق [Q1] فوراً.
# """


# QUERY_EXPANSION_PROMPT = """
# مهمتك: توليد صيغ بحث بديلة لاستعلام قانوني سوري.

# ══════════════════════════════════════
# قواعد صارمة
# ══════════════════════════════════════
# - الإخراج: JSON فقط — بلا مقدمة، بلا شرح، بلا markdown.
# - اللغة: العربية الفصحى الحديثة حصراً — ممنوع منعاً باتاً أي حرف لاتيني.
# - العدد: 3 صيغ بديلة بالضبط — لا أقل، لا أكثر.
# - كل صيغة يجب أن تكون مختلفة لغوياً (مرادفات، توسيع، تضييق).
# - لا تخترع مصطلحات غير موجودة في التشريع السوري.

# ══════════════════════════════════════
# نموذج الإخراج المطلوب
# ══════════════════════════════════════
# {{
#   "query_variations": [
#     "الصيغة الأولى — توسيع المصطلح",
#     "الصيغة الثانية — مرادف قانوني",
#     "الصيغة الثالثة — تخصيص القانون"
#   ]
# }}

# ══════════════════════════════════════
# الاستعلام: {query}
# """


# ROUTER_PROMPT = """
# مهمتك: تصنيف الاستعلام القانوني السوري إلى المسار الصحيح.

# ══════════════════════════════════════
# المسارات المتاحة
# ══════════════════════════════════════
# - civil          → القانون المدني، العقود، الملكية، التعويضات
# - penal          → قانون العقوبات، الجرائم، المسؤولية الجزائية
# - personal_status → الأحوال الشخصية، الزواج، الطلاق، الإرث، الحضانة
# - all            → إذا تقاطع الاستعلام بين مسارين أو أكثر بوضوح

# ══════════════════════════════════════
# متى تستخدم "clarify"؟
# ══════════════════════════════════════
# استخدم route = "clarify" فقط إذا:
#   - يحتمل الاستعلام تصنيفين متعارضين تماماً
#   - غياب معلومة حرجة يُغيّر المسار كلياً (مثال: جنسية الطرف، نوع العقد)

# ══════════════════════════════════════
# قواعد الإخراج
# ══════════════════════════════════════
# - JSON فقط — بلا أي نص خارج الـ JSON.
# - المفاتيح الإلزامية: route، reason، clarifying_question
# - clarifying_question = null إذا لم يكن التصنيف "clarify"

# ══════════════════════════════════════
# نموذج الإخراج
# ══════════════════════════════════════
# {{
#   "route": "civil",
#   "reason": "الاستعلام يتعلق بعقد بيع عقار",
#   "clarifying_question": null
# }}

# ══════════════════════════════════════
# الاستعلام: {query}
# """


# QUERY_REFINER_PROMPT = """
# مهمتك: تحويل إدخال المستخدم إلى استعلام قانوني عربي فصيح ودقيق.

# ══════════════════════════════════════
# قواعد المعالجة
# ══════════════════════════════════════

# [R1 — التصحيح اللغوي]
# - صحّح الأخطاء الإملائية والعامية إلى الفصحى القانونية.
# - مثال: "بدي طلق مرتي" → "طلب إيقاع الطلاق"

# [R2 — استخراج جوهر النزاع]
# - إذا كان النص قصة أو شكوى → استخرج النزاع القانوني الجوهري فقط.
# - لا تُضف وقائع غير مذكورة من المستخدم.

# [R3 — تقييم الاكتمال]
# - needs_more_details = true إذا غابت: أطراف النزاع، نوع العقد، الجنسية، أو أي عنصر يُغيّر الحكم القانوني.
# - needs_more_details = false إذا كانت المعلومات كافية للبحث.

# ══════════════════════════════════════
# قواعد الإخراج
# ══════════════════════════════════════
# - JSON فقط — بلا مقدمة أو شرح.
# - اللغة: عربية فصحى حصراً.
# - المفاتيح الإلزامية: refined_query، legal_core، needs_more_details

# ══════════════════════════════════════
# نموذج الإخراج
# ══════════════════════════════════════
# {{
#   "refined_query": "مطالبة بالتعويض عن ضرر ناجم عن إخلال بعقد مقاولة",
#   "legal_core": "المسؤولية العقدية وأحكام التعويض",
#   "needs_more_details": false
# }}

# ══════════════════════════════════════
# الإدخال: {query}
# """

# class AgenticRAGPipeline:
#     def __init__(self, retriever: HybridRetriever, reranker: LegalReranker):
#         self.retriever = retriever
#         self.reranker = reranker

#     @staticmethod
#     def _extract_json(raw: str) -> dict[str, Any]:
#         decoder = json.JSONDecoder()
#         for start_idx, char in enumerate(raw):
#             if char != "{":
#                 continue
#             try:
#                 parsed, _ = decoder.raw_decode(raw[start_idx:])
#                 if isinstance(parsed, dict):
#                     return parsed
#             except Exception:
#                 continue
#         return {}

#     @staticmethod
#     def _clean_retrieved_text(text: str) -> str:
#         if not text:
#             return ""
#         cleaned = text.replace("\r", "\n")
#         cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
#         noisy_patterns = [
#             r"^الصفحة\s+\d+.*$",
#             r"^Page\s+\d+.*$",
#             r"^جميع الحقوق محفوظة.*$",
#             r"^تم التحميل من.*$",
#             r"^هذا النص للاطلاع.*$",
#         ]
#         lines = []
#         for line in cleaned.split("\n"):
#             ln = line.strip()
#             if not ln:
#                 continue
#             if any(re.match(p, ln, flags=re.IGNORECASE) for p in noisy_patterns):
#                 continue
#             lines.append(ln)
#         return "\n".join(lines).strip()

#     @staticmethod
#     def _format_context(chunks: list[dict[str, Any]]) -> str:
#         parts = []
#         for c in chunks[:2]:
#             meta = c.get("metadata", {})
#             art = meta.get("article_number", "?")
#             law = meta.get("law_name", "القانون السوري")
#             short_text = AgenticRAGPipeline._clean_retrieved_text(c.get("text", ""))[:700]
#             parts.append(f"بناءً على [{law}] - [المادة {art}]:\n{short_text}")
#         return "\n\n".join(parts)

#     @staticmethod
#     def _build_retrieved_law_section(chunks: list[dict[str, Any]], limit: int = 2) -> str:
#         """Create a mandatory section with exact retrieved legal text."""
#         if not chunks:
#             return "نص القانون المسترجع:\nلم يتم العثور على نصوص قانونية."
#         section_lines: list[str] = ["نص القانون المسترجع:"]
#         for chunk in chunks[:limit]:
#             meta = chunk.get("metadata", {})
#             article = meta.get("article_number", "?")
#             raw_text = AgenticRAGPipeline._clean_retrieved_text(
#                 (chunk.get("page_content", chunk.get("text", "")) or "").strip()
#             )
#             section_lines.append(f"- المادة {article}: {raw_text}")
#         return "\n".join(section_lines)

#     def expand_query(self, question: str) -> list[str]:
#         def _run_refiner(payload: dict[str, str]) -> list[str]:
#             normalized_input_query = clean_text(payload["query"], apply_reversal_fix=False)
#             response = chat_completion(
#                 [
#                     {"role": "system", "content": "You produce strict JSON only."},
#                     {"role": "user", "content": QUERY_REFINER_PROMPT.format(query=normalized_input_query)},
#                 ],
#                 temperature=0.0,
#             )
#             parsed = self._extract_json(response)
#             refined = parsed.get("refined_query") or parsed.get("legal_core")
#             refined_query = clean_text(str(refined) if refined else normalized_input_query, apply_reversal_fix=False)
#             # Expand into 3 alternative legal search queries.
#             fallback = chat_completion(
#                 [
#                     {"role": "system", "content": "You produce strict JSON only."},
#                     {"role": "user", "content": QUERY_EXPANSION_PROMPT.format(query=refined_query)},
#                 ],
#                 temperature=0.0,
#             )
#             fallback_parsed = self._extract_json(fallback)
#             variations = fallback_parsed.get("query_variations", [])
#             if isinstance(variations, str):
#                 variations = [variations]
#             if not isinstance(variations, list):
#                 variations = []
#             cleaned = [clean_text(str(v).strip(), apply_reversal_fix=False) for v in variations if str(v).strip()]
#             # Ensure at least one query always exists.
#             unique_queries = []
#             for q in [refined_query, *cleaned]:
#                 if q and q not in unique_queries:
#                     unique_queries.append(q)
#             return unique_queries[:3]

#         if RunnablePassthrough is not None:
#             chain = RunnablePassthrough.assign(refined_query=lambda x: _run_refiner(x))
#             out = chain.invoke({"query": question})
#             return out.get("refined_query", [question])
#         return _run_refiner({"query": question})

#     def decide_route(self, question: str) -> dict[str, str]:
#         response = chat_completion(
#             [
#                 {"role": "system", "content": "You are a legal routing controller. Output JSON only."},
#                 {"role": "user", "content": ROUTER_PROMPT.format(query=question)},
#             ],
#             temperature=0.0,
#         )
#         parsed = self._extract_json(response)
#         route = parsed.get("route", "all")
#         if route not in {"civil", "penal", "personal_status", "all", AMBIGUITY_ROUTE}:
#             route = "all"
#         return {
#             "route": route,
#             "reason": parsed.get("reason", ""),
#             "clarifying_question": parsed.get("clarifying_question", "هل يمكنك توضيح نوع القضية بشكل أدق؟"),
#         }

#     def retrieve_for_query(self, expanded_query: list[str] | str, route: str) -> list[dict[str, Any]]:
#         normalized_query = (
#             [clean_text(q, apply_reversal_fix=False) for q in expanded_query]
#             if isinstance(expanded_query, list)
#             else clean_text(expanded_query, apply_reversal_fix=False)
#         )
#         effective_route = "all" if route == "all" else route
#         results = self.retriever.retrieve(normalized_query, database_route=effective_route)
#         if not isinstance(results, list):
#             return []
#         return [r for r in results if isinstance(r, dict)]

#     @staticmethod
#     def _is_no_result_answer(answer: str) -> bool:
#         normalized = (answer or "").strip().lower()
#         markers = [
#             "i don't know",
#             "i do not know",
#             "no results found",
#             "لم أجد",
#             "لا أعرف",
#         ]
#         return any(marker in normalized for marker in markers)

#     def synthesize_answer(self, question: str, relevant: list[dict[str, Any]]) -> tuple[str, list[int]]:
#         if not relevant:
#             return (
#                 "بصفتي محامياً سورياً، لم أجد مادة مطابقة في النتائج الحالية. "
#                 "قد يكون النص القانوني في باب آخر من القانون السوري. "
#                 "يرجى تزويدي بتفاصيل إضافية عن الواقعة (الزمان، المكان، الأطراف، والنتيجة)."
#             ), []
#         context = self._format_context(relevant)
#         prompt = LEGAL_SYNTH_PROMPT.format(retrieved_articles=context, user_question=question)
#         answer = chat_completion(
#             [
#                     {"role": "system", "content": "You are an expert Syrian Legal Consultant. Provide professional, empathetic Modern Standard Arabic legal advice using only retrieved articles from Syrian Penal, Personal Status, and Civil laws."},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0.0,
#         )
#         if self._is_no_result_answer(answer):
#             answer = (
#                 "تعذّر الوصول إلى نتيجة قانونية دقيقة من السياق الحالي. "
#                 "يرجى تزويدي بوقائع إضافية (الأطراف، التاريخ، نوع النزاع) لرفع دقة الاسترجاع."
#             )
#         law_section = self._build_retrieved_law_section(relevant)
#         if "نص القانون المسترجع" not in answer:
#             answer = f"{answer}\n\n{law_section}"
#         citations = sorted({int(m.group(1)) for m in re.finditer(r"(?:المادة|Article)\s*(\d+)", answer, re.IGNORECASE)})
#         if not citations:
#             answer = (
#                 answer
#                 + "\n\nلم تظهر مادة قانونية محددة في النتائج الحالية. "
#                 "يرجى توضيح الوقائع (الزمن/المكان/الأطراف) للحصول على مواد أدق."
#             )
#         return answer, citations

#     def run(self, question: str) -> dict[str, Any]:
#         stage_times: dict[str, float] = {}
#         started = time.perf_counter()

#         t0 = time.perf_counter()
#         expanded = self.expand_query(question)
#         stage_times["query_expansion"] = time.perf_counter() - t0

#         t0 = time.perf_counter()
#         route_decision = self.decide_route(question)
#         stage_times["route_decision"] = time.perf_counter() - t0

#         if route_decision["route"] == AMBIGUITY_ROUTE:
#             return {
#                 "answer": route_decision["clarifying_question"],
#                 "is_clarification": True,
#                 "route": route_decision["route"],
#                 "route_reason": route_decision["reason"],
#                 "expanded_queries": expanded,
#                 "relevant_articles": [],
#                 "citations": [],
#                 "stage_times": stage_times,
#                 "total_latency": time.perf_counter() - started,
#             }

#         t0 = time.perf_counter()
#         candidates = self.retrieve_for_query(expanded, route_decision["route"])
#         stage_times["retrieval"] = time.perf_counter() - t0

#         t0 = time.perf_counter()
#         relevant = self.reranker.rerank(question, candidates)
#         if not relevant and candidates:
#             # Safety fallback: if reranker yields nothing, keep top retrieved chunks.
#             relevant = candidates[:2]
#         stage_times["reranking"] = time.perf_counter() - t0

#         t0 = time.perf_counter()
#         answer, citations = self.synthesize_answer(question, relevant)
#         stage_times["synthesis"] = time.perf_counter() - t0
#         no_result = self._is_no_result_answer(answer) or not relevant

#         return {
#             "answer": answer,
#             "is_clarification": False,
#             "route": route_decision["route"],
#             "route_reason": route_decision["reason"],
#             "expanded_query": expanded[0] if isinstance(expanded, list) and expanded else str(expanded),
#             "expanded_queries": expanded if isinstance(expanded, list) else [str(expanded)],
#             "relevant_articles": relevant,
#             "citations": citations,
#             "no_result": no_result,
#             "evaluation_contexts": [a.get("text", "") for a in relevant if a.get("text")],
#             "stage_times": stage_times,
#             "total_latency": time.perf_counter() - started,
#         }

"""
Agentic RAG Pipeline for Syrian Legal AI.
Optimized: Combined Refine+Route in one LLM call, faster context formatting.
"""

import json
import re
import time
from typing import Any

try:
    from config import AMBIGUITY_ROUTE
    from ollama_client import chat_completion
    from retriever import HybridRetriever
    from reranker import LegalReranker
    from text_normalization import clean_text
except ModuleNotFoundError:
    from config import AMBIGUITY_ROUTE
    from ollama_client import chat_completion
    from retriever import HybridRetriever
    from reranker import LegalReranker
    from text_normalization import clean_text

try:
    from langchain_core.runnables import RunnablePassthrough
except Exception:
    RunnablePassthrough = None


# ============================================================
# SYRIAN LEGAL AI — PROMPT ENGINEERING v2.0
# Anti-Hallucination Architecture
# ============================================================

LEGAL_SYNTH_PROMPT = """
أنت: مستشار قانوني سوري دقيق ومتخصص. مهمتك تقديم إجابات قانونية مبنية **حصراً** على النصوص المسترجعة.

══════════════════════════════════════
🔴 قواعد صارمة — لا استثناء
══════════════════════════════════════

[Q1 — قاعدة الأدلة الحديدية]
- إذا كانت المواد المسترجعة لا تتعلق بسؤال المستخدم بشكل مباشر:
  → لا تفسّر، لا تخمّن، لا تستنتج بعيداً عن النص.
  → أجب: "عذراً، لم أعثر في قاعدة البيانات على نصوص قانونية دقيقة تغطي هذه المسألة. أنصح بمراجعة محامٍ مختص."

[Q2 — قاعدة الاقتباس الحرفي]
- كل مادة قانونية تذكرها يجب أن:
  ✓ تُدرج رقمها الصريح (مثال: المادة 386 من القانون المدني)
  ✓ تقتبس نصها كما ورد في السياق بين علامتي تنصيص « »
  ✗ لا تُعدّل الصياغة القانونية أبداً

[Q3 — قاعدة عدم الاختراع]
- محظور: ذكر أرقام مواد أو قوانين غير موجودة في السياق المقدّم.
- إذا لم يحدد السياق رقم المادة → اكتب: "نص قانوني بدون رقم محدد في السياق"

[Q4 — قاعدة الوضوح]
- الإجابة الكاملة لا تتجاوز 200 كلمة.
- لا رموز أجنبية، لا Latin characters.

══════════════════════════════════════
📋 هيكل الإجابة الإلزامي
══════════════════════════════════════

**النتيجة:** [جملة واحدة مباشرة تجيب السؤال]

**السند القانوني:**
- [رقم المادة والقانون]: «نص الاقتباس من السياق»

**تحفظات جوهرية:** [فقط إذا وُجد تعارض أو شرط في النص]

**سؤال توضيحي:** [سؤال واحد فقط إذا كانت التفاصيل ناقصة]

══════════════════════════════════════
📥 المدخلات
══════════════════════════════════════
السؤال: {user_question}

السياق القانوني المسترجع:
{retrieved_articles}

⚠️ تذكّر: إذا لم يغطِّ السياق السؤالَ → طبّق [Q1] فوراً.
"""

# ✅ برومبت موحّد يدمج Refiner + Router في استدعاء واحد
COMBINED_REFINE_ROUTE_PROMPT = """
مهمتك: تنقية الاستعلام القانوني وتصنيفه في خطوة واحدة.

══════════════════════════════════════
خطوات المعالجة
══════════════════════════════════════
[1] نقّح الاستعلام: صحّح الأخطاء الإملائية والعامية، استخرج جوهر النزاع، لا تخترع وقائع.
[2] صنّفه إلى أحد المسارات:
    - civil          → القانون المدني، العقود، الملكية، التعويضات
    - penal          → قانون العقوبات، الجرائم، المسؤولية الجزائية
    - personal_status → الزواج، الطلاق، الإرث، الحضانة
    - all            → يتقاطع مسارين أو أكثر
    - clarify        → غامض أو تنقصه معلومة حرجة تُغيّر المسار

══════════════════════════════════════
قواعد الإخراج — JSON فقط بلا أي نص خارجه
══════════════════════════════════════
{{
  "refined_query": "الاستعلام المُنقّح بالفصحى القانونية",
  "legal_core": "جوهر النزاع القانوني",
  "needs_more_details": false,
  "route": "civil",
  "route_reason": "سبب التصنيف",
  "clarifying_question": null
}}

الإدخال: {query}
"""

QUERY_EXPANSION_PROMPT = """
مهمتك: توليد صيغ بحث بديلة لاستعلام قانوني سوري.

قواعد صارمة:
- الإخراج: JSON فقط — بلا مقدمة، بلا شرح.
- اللغة: العربية الفصحى حصراً.
- العدد: 3 صيغ بديلة بالضبط.
- كل صيغة مختلفة لغوياً (مرادفات، توسيع، تضييق).

{{
  "query_variations": ["...", "...", "..."]
}}

الاستعلام: {query}
"""


class AgenticRAGPipeline:
    def __init__(self, retriever: HybridRetriever, reranker: LegalReranker):
        self.retriever = retriever
        self.reranker  = reranker
    def decide_route(self, question: str) -> dict[str, str]:
        result = self.refine_and_route(question)
        return {
            "route": result["route"],
            "reason": result["route_reason"],
            "clarifying_question": result["clarifying_question"] or "هل يمكنك التوضيح؟"
        }
    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for start_idx, char in enumerate(raw):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(raw[start_idx:])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}

    @staticmethod
    def _clean_retrieved_text(text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\r", "\n")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        noisy_patterns = [
            r"^الصفحة\s+\d+.*$",
            r"^Page\s+\d+.*$",
            r"^جميع الحقوق محفوظة.*$",
            r"^تم التحميل من.*$",
            r"^هذا النص للاطلاع.*$",
        ]
        lines = []
        for line in cleaned.split("\n"):
            ln = line.strip()
            if not ln:
                continue
            if any(re.match(p, ln, flags=re.IGNORECASE) for p in noisy_patterns):
                continue
            lines.append(ln)
        return "\n".join(lines).strip()

    @staticmethod
    def _format_context(chunks: list[dict[str, Any]]) -> str:
        parts = []
        for c in chunks[:3]:
            meta       = c.get("metadata", {})
            art        = meta.get("article_number", "?")
            law        = meta.get("law_name", "القانون السوري")
            short_text = AgenticRAGPipeline._clean_retrieved_text(
                c.get("text", "")
            )[:1000]
            parts.append(f"- المادة {art} [{law}]:\n{short_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_retrieved_law_section(chunks: list[dict[str, Any]], limit: int = 3) -> str:
        if not chunks:
            return "نص القانون المسترجع:\nلم يتم العثور على نصوص قانونية."
        section_lines: list[str] = ["نص القانون المسترجع:"]
        for chunk in chunks[:limit]:
            meta     = chunk.get("metadata", {})
            article  = meta.get("article_number", "?")
            raw_text = AgenticRAGPipeline._clean_retrieved_text(
                (chunk.get("page_content", chunk.get("text", "")) or "").strip()
            )
            section_lines.append(f"- المادة {article}: {raw_text}")
        return "\n".join(section_lines)

    def refine_and_route(self, question: str) -> dict[str, Any]:
        normalized = clean_text(question, apply_reversal_fix=False)
        response = chat_completion(
            [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": COMBINED_REFINE_ROUTE_PROMPT.format(
                    query=normalized
                )},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        parsed = self._extract_json(response)

        refined = parsed.get("refined_query") or normalized
        route   = parsed.get("route", "all")
        if route not in {"civil", "penal", "personal_status", "all", "clarify"}:
            route = "all"

        return {
            "refined_query":       refined,
            "legal_core":          parsed.get("legal_core", ""),
            "needs_more_details":  parsed.get("needs_more_details", False),
            "route":               route,
            "route_reason":        parsed.get("route_reason", ""),
            "clarifying_question": parsed.get("clarifying_question"),
        }

    @staticmethod
    def _simple_expand(query: str) -> list[str]:
        """توسيع بسيط بدون LLM — مرادفات قانونية شائعة"""
        synonyms = {
            "بيع":     ["عقد البيع", "البيع والشراء"],
            "إيجار":   ["عقد الإيجار", "تأجير العقار"],
            "طلاق":    ["فسخ عقد الزواج", "إيقاع الطلاق"],
            "إرث":     ["الميراث", "التركة والورثة"],
            "تعويض":   ["المسؤولية المدنية", "الضرر والتعويض"],
            "سرقة":    ["جريمة السرقة", "الاستيلاء على المال"],
            "قتل":     ["جريمة القتل", "إزهاق الروح"],
            "عقد":     ["الالتزامات التعاقدية", "أركان العقد"],
            "ملكية":   ["حق الملكية", "التملك العقاري"],
            "حضانة":   ["حق الحضانة", "حضانة الأطفال"],
        }
        variations = [query]
        for key, vals in synonyms.items():
            if key in query:
                variations.extend(vals)
                break
        return list(dict.fromkeys(variations))[:3]

    def expand_query(self, refined_query: str) -> list[str]:
        response = chat_completion(
            [
                {"role": "system", "content": "You produce strict JSON only."},
                {"role": "user", "content": QUERY_EXPANSION_PROMPT.format(
                    query=refined_query
                )},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        parsed     = self._extract_json(response)
        variations = parsed.get("query_variations", [])
        if not isinstance(variations, list):
            variations = []
        cleaned = [
            clean_text(str(v).strip(), apply_reversal_fix=False)
            for v in variations if str(v).strip()
        ]
        unique = []
        for q in [refined_query, *cleaned]:
            if q and q not in unique:
                unique.append(q)
        return unique[:3]

    def retrieve_for_query(
        self, expanded_query: list[str] | str, route: str
    ) -> list[dict[str, Any]]:
        normalized_query = (
            [clean_text(q, apply_reversal_fix=False) for q in expanded_query]
            if isinstance(expanded_query, list)
            else clean_text(expanded_query, apply_reversal_fix=False)
        )
        effective_route = "all" if route == "all" else route
        results = self.retriever.retrieve(normalized_query, database_route=effective_route)
        if not isinstance(results, list):
            return []
        return [r for r in results if isinstance(r, dict)]

    @staticmethod
    def _is_no_result_answer(answer: str) -> bool:
        normalized = (answer or "").strip().lower()
        markers = [
            "i don't know", "i do not know", "no results found",
            "لم أجد", "لا أعرف",
        ]
        return any(marker in normalized for marker in markers)

    def synthesize_answer(
        self, question: str, relevant: list[dict[str, Any]]
    ) -> tuple[str, list[int]]:
        if not relevant:
            return (
                "بصفتي محامياً سورياً، لم أجد مادة مطابقة في النتائج الحالية. "
                "يرجى تزويدي بتفاصيل إضافية عن الواقعة (الزمان، المكان، الأطراف)."
            ), []

        context = self._format_context(relevant)
        prompt  = LEGAL_SYNTH_PROMPT.format(
            retrieved_articles=context,
            user_question=question,
        )
        answer = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "أنت مستشار قانوني سوري متخصص. "
                        "قدّم إجابات قانونية دقيقة بالعربية الفصحى "
                        "مستنداً فقط إلى النصوص القانونية المُسترجعة."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        if self._is_no_result_answer(answer):
            answer = (
                "تعذّر الوصول إلى نتيجة قانونية دقيقة من السياق الحالي. "
                "يرجى تزويدي بوقائع إضافية (الأطراف، التاريخ، نوع النزاع)."
            )

        law_section = self._build_retrieved_law_section(relevant)
        if "نص القانون المسترجع" not in answer:
            answer = f"{answer}\n\n{law_section}"

        citations = sorted({
            int(m.group(1))
            for m in re.finditer(r"(?:المادة|Article)\s*(\d+)", answer, re.IGNORECASE)
        })
        if not citations:
            answer += (
                "\n\nلم تظهر مادة قانونية محددة في النتائج الحالية. "
                "يرجى توضيح الوقائع للحصول على مواد أدق."
            )

        return answer, citations

    def run(self, question: str) -> dict[str, Any]:
        stage_times: dict[str, float] = {}
        started = time.perf_counter()

        t0 = time.perf_counter()
        combined = self.refine_and_route(question)
        stage_times["refine_and_route"] = time.perf_counter() - t0

        refined_query = combined["refined_query"]
        route         = combined["route"]

        if route == AMBIGUITY_ROUTE:
            return {
                "answer":           combined["clarifying_question"] or "هل يمكنك توضيح نوع القضية؟",
                "is_clarification": True,
                "route":            route,
                "route_reason":     combined["route_reason"],
                "expanded_queries": [refined_query],
                "relevant_articles":[],
                "citations":        [],
                "stage_times":      stage_times,
                "total_latency":    time.perf_counter() - started,
            }

        expanded = self._simple_expand(refined_query)

        t0 = time.perf_counter()
        candidates = self.retrieve_for_query(expanded, route)
        stage_times["retrieval"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        relevant = self.reranker.rerank(question, candidates)
        if not relevant and candidates:
            relevant = candidates[:3]
        stage_times["reranking"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        answer, citations = self.synthesize_answer(question, relevant)
        stage_times["synthesis"] = time.perf_counter() - t0

        return {
            "answer":            answer,
            "is_clarification":  False,
            "route":             route,
            "route_reason":      combined["route_reason"],
            "expanded_query":    refined_query,
            "expanded_queries":  expanded,
            "relevant_articles": relevant,
            "citations":         citations,
            "no_result":         not relevant,
            "evaluation_contexts": [
                a.get("text", "") for a in relevant if a.get("text")
            ],
            "stage_times":    stage_times,
            "total_latency":  time.perf_counter() - started,
        }