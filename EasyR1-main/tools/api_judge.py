"""
Answer judging module with VLMEvalKit-inspired matching logic.

Three-tier pipeline:
  Level 1 (Fast Path): Rule-based matching — exact match, option extraction, VQA normalization
  Level 2 (Slow Path):  API-based semantic judging — only for cases Level 1 cannot resolve
"""

from typing import List, Optional, Dict, Tuple
from tools.custom_api import get_api_response
import traceback
import time
import re
import string
import copy as cp
import os


# ============================================================================
# Section 1: Answer Extraction (inspired by VLMEvalKit matching_util.py)
# ============================================================================

def extract_answer_from_tags(text: str) -> Optional[str]:
    """Extract the last answer from <answer>...</answer> tags."""
    if not isinstance(text, str):
        return None
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def can_infer_option(answer: str, choices: dict) -> Optional[str]:
    """
    Try to extract an option letter (A/B/C/D/...) from the answer text.
    Adapted from VLMEvalKit's can_infer_option.
    """
    if not isinstance(answer, str):
        return None

    # Reject known error messages
    reject_phrases = [
        "Failed to obtain answer via API",
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for phrase in reject_phrases:
        if phrase in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    # Remove punctuation that might interfere with tokenization
    for ch in '.()[],:;!*#{}':
        answer_mod = answer_mod.replace(ch, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            # Option letter must appear near the end of the answer
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return None


def can_infer_text(answer: str, choices: dict) -> Optional[str]:
    """
    Try to match the answer against option text content.
    Adapted from VLMEvalKit's can_infer_text.

    NOTE: VLMEvalKit's original has a length guard (answer > 2x total option length)
    that causes false negatives for short options like "red"/"blue" in long answers.
    We relax this: only apply the guard when answer > 4x total option length,
    since we're already extracting <answer> tags before calling this.
    """
    if not isinstance(answer, str):
        return None
    answer_lower = answer.lower()

    # Relaxed length guard compared to VLMEvalKit's 2x
    total_opt_len = sum(len(str(v)) for v in choices.values())
    if total_opt_len > 0 and len(answer_lower) > 4 * total_opt_len:
        return None

    choices_lower = {k: str(v).lower() for k, v in choices.items()}
    cands = []
    for k, v in choices_lower.items():
        if v and v in answer_lower:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return None


def can_infer(answer: str, choices: dict) -> Optional[str]:
    """
    Two-stage option extraction: first try option letter, then option text.
    Adapted from VLMEvalKit's can_infer.
    """
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    if copt:
        return copt
    return can_infer_text(answer, choices)


def build_choices_from_question(question: str) -> dict:
    """
    Parse option letters and content from a question string.
    Supports formats like:
      "A. xxx  B. yyy  C. zzz  D. www"
      "A) xxx B) yyy ..."
      "(A) xxx (B) yyy ..."
    """
    if not isinstance(question, str):
        return {}
    choices = {}
    # Match patterns like "A. xxx", "A) xxx", "(A) xxx"
    pattern = r'(?:^|\n|\s)\(?([A-Z])\)?[.\s)]\s*(.*?)(?=\s+\(?[A-Z]\)?[.\s)]|$)'
    matches = re.findall(pattern, question, re.DOTALL)
    for letter, text in matches:
        if letter in string.ascii_uppercase:
            choices[letter] = text.strip()
    return choices


# ============================================================================
# Section 2: VQA-style Answer Normalization (from VLMEvalKit vqa_eval.py)
# ============================================================================

DIGIT_MAP = {
    'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7',
    'eight': '8', 'nine': '9', 'ten': '10',
}

ARTICLES = {'a', 'an', 'the'}


def process_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return re.sub(r'[^\w\s]', ' ', text)


def normalize_vqa_answer(text: str) -> str:
    """
    Normalize an answer for VQA-style comparison:
    - lowercase
    - convert number words to digits ("two" -> "2")
    - remove articles ("a", "an", "the")
    - remove punctuation
    - normalize whitespace
    """
    if not isinstance(text, str):
        return ''
    words = text.lower().split()
    out = []
    for w in words:
        w = DIGIT_MAP.get(w, w)
        if w not in ARTICLES:
            out.append(w)
    text = ' '.join(out)
    text = process_punctuation(text)
    text = ' '.join(text.split())  # normalize whitespace
    return text.strip()


# ============================================================================
# Section 3: Unified Rule-Based Matching (Fast Path)
# ============================================================================

def rule_based_match(pred: str, gt: str, question: Optional[str] = None) -> Optional[bool]:
    """
    Comprehensive rule-based matching combining:
    1. <answer> tag extraction + exact match
    2. Option letter extraction (can_infer) for MCQ questions
    3. VQA-style normalized match
    4. Simple string match after stripping

    Returns:
        True  — definitely correct
        False — definitely incorrect
        None  — cannot determine, needs API judge
    """
    if pred is None or gt is None:
        return False

    # --- Step 1: Extract answers from <answer> tags ---
    gt_answer = extract_answer_from_tags(gt)
    if gt_answer is None:
        gt_answer = gt.strip()

    pred_answer = extract_answer_from_tags(pred)
    if pred_answer is None:
        # No <answer> tag in prediction — try to extract from the raw output
        pred_answer = pred.strip()

    gt_lower = gt_answer.lower().strip()
    pred_lower = pred_answer.lower().strip()

    # --- Step 2: Exact match ---
    if gt_lower == pred_lower:
        return True

    # --- Step 3: MCQ option extraction ---
    # If GT is a single letter (A/B/C/D), this is likely a multiple-choice question
    if len(gt_lower) == 1 and gt_lower.isalpha():
        # 3a: If pred has <answer> tag content, try matching "A. xxx" format
        match_prefix = re.match(r'^([a-z])\.\s', pred_lower)
        if match_prefix and match_prefix.group(1) == gt_lower:
            return True

        # 3b: Try can_infer on the full prediction text (handles "The answer is A", etc.)
        choices = {}
        if question:
            choices = build_choices_from_question(question)
        if choices:
            # Ensure GT letter is in choices
            if gt_lower.upper() in choices or gt_lower in [c.lower() for c in choices]:
                inferred = can_infer(pred, choices)
                if inferred:
                    return inferred.lower() == gt_lower

        # 3c: Fallback — scan for the option letter near the end of the answer
        # This handles cases like "I choose A" or "So the answer is B"
        pred_clean = pred
        for ch in '.()[],:;!*#{}':
            pred_clean = pred_clean.replace(ch, ' ')
        splits = [x.strip() for x in pred_clean.split()]
        if len(splits) > 0:
            letter_count = sum(1 for s in splits if s.lower() == gt_lower and len(s) == 1)
            if letter_count == 1:
                # Make sure the letter appears near the end
                for idx in range(len(splits)):
                    if splits[idx].lower() == gt_lower and len(splits[idx]) == 1:
                        if idx >= len(splits) - 5:
                            return True

    # --- Step 4: VQA-style normalized match ---
    gt_norm = normalize_vqa_answer(gt_answer)
    pred_norm = normalize_vqa_answer(pred_answer)
    if gt_norm and pred_norm and gt_norm == pred_norm:
        return True

    # --- Step 5: If prediction is very long (contains reasoning), we can't be sure ---
    # If no <answer> tag and the prediction is much longer than GT,
    # we can't confidently say it's wrong — delegate to API
    if extract_answer_from_tags(pred) is None and len(pred_lower) > 3 * max(len(gt_lower), 1):
        return None  # Cannot determine, need API

    # --- Step 6: Simple strip + punctuation removal match ---
    if process_punctuation(gt_lower).strip() == process_punctuation(pred_lower).strip():
        return True

    return None


# ============================================================================
# Section 4: API-based Semantic Judging (Slow Path)
# ============================================================================

def judge_wrap_fn(pred: Optional[str], gt: Optional[str], question: Optional[str],
                  repetition_penalty: bool = False) -> Tuple[str, str]:
    sys_prompt = """
# 角色
你是一个答案一致性判断专家。你的唯一任务是判断[模型回答]与[参考答案]是否语义一致。

# 任务规范
你不需要知道问题真正的正确答案，只需判断[模型回答]所表达的含义是否与[参考答案]一致。注意：
1. 忽略[模型回答]中的推理过程、分析说明、格式标签（如<answer>）等内容，只关注其最终给出的结论性答案。
2. 如果[模型回答]和[参考答案]使用了不同的语法结构或词语，但指向同一个结论，应判为一致。
3. 如果[模型回答]的逻辑方向与[参考答案]相反（如"左边" vs "右边"、"Yes" vs "No"），无论解释多么合理，都应判为不一致。

# 判定标准

## 标准一：图片问答/描述类（问题含<image>占位符或无选项）
适用于以 "Is the ... on the left/right?"、"What is the color of...?" 等形式提问，答案通常为完整描述句。
- 若[参考答案]为 "Yes/No + 陈述"（如 "Yes, the cat is on the left."），[模型回答]只需在逻辑方向上匹配（Yes/No 或
left/right）即可；具体修饰性词语差异可忽略。
- 若[参考答案]为纯粹描述（如 "The text is ABC."），[模型回答]的核心事实必须与[参考答案]相同。

## 标准二：选择题类（问题含A. / B. / C. / D.选项）
适用于题目中明确给出若干选项的情况。
- [参考答案]可能以字母标记（如 "A"）、字母加具体内容（如 "A. 具体内容"）或仅具体内容形式出现。
- [模型回答]若仅给出字母（如 "A"），需与[参考答案]的字母标记一致。
- [模型回答]若给出具体内容（如 "具体内容"），需与[参考答案]中对应字母标记的选项内容语义一致。
- 若[模型回答]给出类似 "A. 具体内容" 的混合形式，只要字母和内容的对应关系与[参考答案]一致即可。
- 可忽略大小写、中英文标点、LaTeX格式差异。

# 输出格式
你必须在最后给出结论，且严格使用以下格式：
<最终结果>
\\boxed{Yes}
</最终结果>
或
<最终结果>
\\boxed{No}
</最终结果>

以下是输入内容：
"""
    user_prompt = (
        f"[问题]: {question if question is not None else ''}\n"
        f"[模型回答]: {pred if pred is not None else ''}\n"
        f"[参考答案]: {gt if gt is not None else ''}\n"
    )
    return sys_prompt, user_prompt


def _parse_api_judgment(response: str) -> Optional[bool]:
    """
    Parse the API judge's response to determine Yes/No.
    Returns True (correct), False (incorrect), or None (ambiguous).
    """
    if not response or not isinstance(response, str):
        return None

    f_response = response.strip()

    # Stage 1: Extract <最终结果> tag (take the last one)
    final_matches = re.findall(r'<最终结果>(.*?)</最终结果>', f_response, re.DOTALL)
    if final_matches:
        f_response = final_matches[-1].strip()

    # Stage 2: Extract \boxed{...} content (take the last one)
    # Handle both \boxed{Yes} and boxed{Yes}
    boxed_matches = re.findall(r'\\?boxed\{(.*?)\}', f_response, re.DOTALL)
    if boxed_matches:
        f_response = boxed_matches[-1].strip()
        
    # Stage 3: Strip punctuation for robust matching
    t = f_response.strip().lower()
    t_clean = t.strip('.,;:!? \n\r\t')

    # Stage 4: Exact word match
    if t_clean == "yes":
        return True
    elif t_clean == "no":
        return False

    # Stage 5: Substring fallback (more lenient)
    # Only use if exactly one of yes/no appears, to avoid ambiguity
    if "yes" in t_clean and "no" not in t_clean:
        return True
    if "no" in t_clean and "yes" not in t_clean:
        return False

    return None  # Ambiguous


def _api_call_wrapper(
    api_name: str,
    pred: Optional[str],
    gt: Optional[str],
    question: Optional[str],
    dataset_name: str,
    client=None,
    api_kwargs: Optional[dict] = None,
    repetition_penalty: bool = False
) -> Optional[float]:
    """
    Execute API-based judging with up to `attempts` retries.
    Retries on both exceptions AND ambiguous responses.
    """
    if pred is None or gt is None or str(pred).strip() == "":
        return False

    attempts = 3
    for atpt in range(attempts):
        try:
            sys_prompt, user_prompt = judge_wrap_fn(pred, gt, question, repetition_penalty)

            responses = get_api_response(
                api_name, sys_prompt, [user_prompt],
                client=client, **(api_kwargs or {})
            )

            if not responses or not isinstance(responses, list) or len(responses) == 0:
                continue
            if not isinstance(responses[0], str) or not responses[0].strip():
                continue

            judgment = _parse_api_judgment(responses[0])

            if judgment is True:
                return 1.0
            elif judgment is False:
                return 0.0
            else:
                # Ambiguous response — retry
                print(f"WARNING: Ambiguous API response (attempt {atpt+1}/{attempts}), retrying... "
                      f"Response: {responses[0][:200]}")
                continue

        except Exception as e:
            traceback.print_exc()
            print(f"API judge error (attempt {atpt+1}/{attempts}): {e}")
            continue

    # All attempts failed to yield a clear response
    return None


# ============================================================================
# Section 5: Utility Functions
# ============================================================================

def _strip_boxed_instruction(q: str) -> str:
    """Remove model instruction from the question before sending to API judge."""
    if not isinstance(q, str):
        return q
    return (
        q.replace("Analyze the image within ​​ tags; if details are unclear, output enhancement visual tokens in <START_OF_GEN> and <END_OF_GEN>; if the image is sufficient, output the final answer in <answer> tags.", "")
        .strip()
    )


# ============================================================================
# Section 6: Batch Judge (Main Entry Point)
# ============================================================================

def api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = 'gemini-2.5-pro',
    api_max_workers: int = 16,
    api_kwargs: Optional[Dict] = None,
    client=None,
    dataset_name: str = "",
    repetition_penalty: bool = False
) -> List[float]:
    """
    Batch judging with rule-based fast path + API slow path.

    For each (question, pred, gt):
      1. Try rule_based_match() — returns True/False/None
      2. If None (cannot determine), call API judge
      3. Fallback to 0.0 on all failures

    Returns:
        List[float]: scores per sample; 1.0 = correct, 0.0 = incorrect.
    """
    import concurrent.futures as cf
    start_time = time.time()

    if not (len(questions) == len(preds) == len(gts)):
        raise ValueError("Length mismatch: `questions`, `preds`, and `gts` must have the same length.")

    n = len(preds)
    results: List[float] = [0.0] * n

    # Strip instruction from questions
    questions_wo_inst = [_strip_boxed_instruction(q) if isinstance(q, str) else q for q in questions]

    try:
        max_workers = int(os.environ.get("API_JUDGE_WORKERS", api_max_workers))
    except Exception:
        max_workers = api_max_workers

    # Phase 1: Rule-based fast path
    api_needed = []  # list of (index, question, pred, gt)
    for i in range(n):
        match_result = rule_based_match(preds[i], gts[i], questions_wo_inst[i])
        if match_result is True:
            results[i] = 1.0
        elif match_result is False:
            results[i] = 0.0
        else:
            # None — cannot determine, need API
            api_needed.append(i)

    if len(api_needed) == 0:
        end_time = time.time()
        print(f"[api_batch_judge] All {n} samples resolved by rule-based matching in {end_time - start_time:.2f}s")
        return results

    # Phase 2: API slow path for undetermined samples
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i in api_needed:
            fut = ex.submit(
                _api_call_wrapper,
                api_name,
                preds[i],
                gts[i],
                questions_wo_inst[i],
                dataset_name,
                client=client,
                api_kwargs=api_kwargs,
                repetition_penalty=repetition_penalty
            )
            futs.append((i, fut))

        for i, fut in futs:
            try:
                r = fut.result()
                results[i] = r if r is not None else 0.0
            except Exception:
                traceback.print_exc()
                print(f"WARNING: API judge fail for index {i}, set to 0.0")
                results[i] = 0.0

    end_time = time.time()
    rule_resolved = n - len(api_needed)
    print(f"[api_batch_judge] Completed {n} samples in {end_time - start_time:.2f}s "
          f"(rule-based: {rule_resolved}, API: {len(api_needed)}, using '{api_name}')")
    return results