
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math
import json
import time
from typing import Dict, List, Union, Optional, Tuple
import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer
from Levenshtein import ratio
from math_verify import parse, verify
#from math_evaluation import is_equiv
import re
import torch
from tools.api_judge import api_batch_judge, rule_based_match, extract_answer_from_tags
from tools.custom_api import get_api_response
import pdb
####################################################################
# rule-based judge
####################################################################


PATTERN_STR = r"<think>.*?</think>\s*<answer>.*?</answer>"
COMPLEX_FORMAT_RE = re.compile(PATTERN_STR, re.DOTALL)

# PATTERN_STR = r"<think>.*?</think>\s*<answer>.*?</answer>"
# COMPLEX_FORMAT_RE = re.compile(PATTERN_STR, re.DOTALL)

def format_reward(predict: str) -> float:
    # 1. 安全检查：处理 None 或 空字符串
    if not predict:
        return 0.0

    # 2. 使用 .search() 扫描整个字符串
    # 只要字符串中包含符合该结构的片段，就返回 True
    if COMPLEX_FORMAT_RE.search(predict):
        return 1.0
    
    return 0.0

def use_latent_reward(predict: str):
    if "<abs_vis_token>" in predict:
        return 1.0
    return 0.0


def extract_and_check(content, sol) -> float:
    """
    Unified rule-based check using api_judge.rule_based_match,
    with math_verify as secondary fallback for numeric answers.
    """
    # Primary: use the unified rule-based matching
    match_result = rule_based_match(content, sol)
    if match_result is True:
        return 1.0
    if match_result is False:
        # Try math_verify as secondary for numeric/mathematical answers
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
        student_answer = content_matches[-1].strip() if content_matches else ''
        if student_answer:
            try:
                answer = parse(student_answer)
                if float(verify(answer, parse(ground_truth))) > 0:
                    return 1.0
            except Exception:
                pass
        return 0.0
    # None — cannot determine rule-based, will be resolved by API judge later
    return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_and_check(response, ground_truth)
    return answer


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1, length_penalty_weight = 0.001, resp_lengths = None, ref_resp_lengths = None) -> List[Dict[str, float]]:
    scores = []
    ref_resp_lengths = torch.tensor(ref_resp_lengths)
    if resp_lengths is not None and ref_resp_lengths is not None:
        length_penalty = torch.where(torch.logical_and(resp_lengths > ref_resp_lengths, ref_resp_lengths!=0), resp_lengths - ref_resp_lengths, torch.zeros_like(resp_lengths))
    else:
        length_penalty = torch.zeros(len(predicts))
    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score - length_penalty_weight * length_penalty[i],
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

def build_prompt_mcq(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)

demo_prompt_extract_and_judge = """
The [Standard Answer] is the correct answer to the question, and the [Model Response] is the answer generated by a model for that question. [Question] is the original question.
Thoroughly read both the [Question], [Standard Answer] and the [Model Response]. You need to:

1. Extract the answer from the [Model Response], output '[Extracted answer]: XXX'.
2. Assess the consistency of the extracted answer with the [Standard Answer] according to the [Question]. If the [Model Answer] is consistent with the [Standard Answer], please output '1'. If not, or the answer for the [Question] cannot be extrated, output '0'.

Below are some examples:
[Question]: A wedding photo of a newlywed couple in front of a castle-like building. What color are the earrings on the bride's ears?
[Standard Answer]: silver
[Model Response]: To answer the question, I need to locate the bride in the image and identify her earrings. The image is quite dark, so I will focus on the bride's face to discern any details on her ears.To get a clearer view of the bride's ears and any accessories, I will generate a zoomed-in image of that specific area.
<START_OF_GEN><IMG_OF_0234><IMG_OF_0966>...<END_OF_GEN>
The zoomed-in view clearly shows the bride's face. Upon close inspection,  her left ear, which is visible in the image, is adorned with  a distinct  red earring.
[Extracted answer]: red
[Judgment]: 0

[Question]: Under the warm yellow candlelight, the two sat opposite each other. The table was piled high with books and scrolls. How many candles were there in total on the table?
[Standard Answer]: Two
[Model Response]: To answer the question, I need to carefully examine the image to locate all the candles present on the table. I will focus on the area around the table where candles might be visible.To accurately count the candles, I will generate a zoomed-in view of the area around the table where candles are typically placed to ensure clear visibility and precise counting.
<START_OF_GEN><IMG_OF_0232><IMG_OF_0066>...<END_OF_GEN>
The zoomed-in image clearly shows 2 distinct candles: one on the left side of the table, one on the right side. Each candle is clearly visible and identifiable.The visual evidence from the detailed view confirms the presence of 2 candles on the table.
[Extracted answer]: 2
[Judgment]: 1

"""



def get_evaluation_chat_response(sys_prompt, user_prompt, client, temperature=0.7):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content


# Check if the judgment is in the correct format
def process_judgment(judgment):
    if judgment is None:
        return False
    judgment = judgment.lower().replace("[judgment]:","").strip()
    if judgment not in ['0', '1']:
        return False
    return True

# Create a test prompt for the model to score the answer
def create_test_prompt(demo_prompt, question, answer, extraction):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"[Question]: {question}\n[Standard Answer]: {answer}\n[Model Response]: {extraction}\n[Extracted answer]: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt



def extract_and_check_api(question: str, predict: str, ground_truth: str, client, verbose=False) -> float:
    sys_prompt = "You are a helper judge assistant."
    retries = 3
    for _ in range(retries):
        try:
            test_prompt = create_test_prompt(demo_prompt_extract_and_judge, question, ground_truth, predict)
            judgment = get_evaluation_chat_response(sys_prompt, test_prompt, client)
            # sometimes gpt may return 'judgment: 1' or 'judgment: 0'
            return process_judgment(judgment)
        except Exception as e:
            print(e, verbose)
            print(f"Error in matching answer:\n[Standard Answer] {ground_truth}\n[Model Answer] {predict}")
    print("All retries failed in extract_and_check_api, fall back to rule-based judge.")
    return extract_and_check(predict, ground_truth)


####################################################################
# Consistency Reward (CSTORM): 推理过程与最终答案的一致性判定
####################################################################

CONSISTENCY_TEMPLATE = '''You are an expert about question answering. I will provide you a solution process and a final answer to the same question. Please evaluate the Consistency between the solution process and the final answer: If the solution process draws the same conclusion with the final answer, output '1'. If not, output '0'.

Here is the solution process and the final answer for you to evaluate:

#### Solution Process:
{gen_solution}

#### Final Answer:
{gen_answer}

Your output ('1' for consistent, '0' for inconsistent):'''


def _cst_extract_thinking(solution: str) -> str:
    """提取  hesitation..thinking 中的推理过程"""
    thinking_pattern = r'<think>(.*?)</think>'
    thinking_match = re.search(thinking_pattern, solution, re.DOTALL)
    thinking = thinking_match.group(1) if thinking_match else ''
    if thinking == "":
        thinking = solution.split("</think>")[0].replace("<think>", "")
    for tag in ["<image>", "<START_OF_GEN>", "<END_OF_GEN>"]:
        thinking = thinking.replace(tag, "")
    return thinking


def _cst_extract_answer(solution: str) -> str:
    """提取 <answer>...</answer> 中的最终答案"""
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, solution, re.DOTALL)
    answer = answer_match.group(1) if answer_match else ''
    return answer.strip()


def _consistency_wrap_fn(predict: str) -> Tuple[str, str]:
    """构建一致性判定的 system/user prompt"""
    thinking = _cst_extract_thinking(predict)[-500:]
    answer = _cst_extract_answer(predict)

    sys_prompt = "You are an expert about question answering. You evaluate whether a solution process and a final answer are consistent."
    user_prompt = CONSISTENCY_TEMPLATE.format(gen_solution=thinking, gen_answer=answer)
    return sys_prompt, user_prompt


def _consistency_api_call_wrapper(
    api_name: str,
    predict: str,
    client=None,
    api_kwargs: Optional[dict] = None,
) -> Optional[float]:
    """
    调用 API 判定推理过程与最终答案的一致性，最多重试3次。
    返回 0.0 / 1.0，失败返回 None。
    """
    answer = _cst_extract_answer(predict)
    if answer == '':
        return 0.0

    sys_prompt, user_prompt = _consistency_wrap_fn(predict)

    max_try = 3
    for _ in range(max_try):
        try:
            responses = get_api_response(
                api_name, sys_prompt, [user_prompt],
                client=client, **(api_kwargs or {})
            )
            # print(f"current user_promp is {user_prompt},  reponse is {responses}")   
            if responses and isinstance(responses[0], str) and responses[0].strip():
                completion = responses[0].strip().lower()
                # 与 api_judge.py 一致的简单解析：检测 '1' 或 '0'
                if "1" in completion and "0" not in completion:
                    return 1.0
                if "0" in completion and "1" not in completion:
                    return 0.0
                # 模糊情况：尝试提取最后一个数字
                nums = re.findall(r'\b([01])\b', completion)
                if nums:
                    return float(nums[-1])
                print(f"[Consistency] Ambiguous response: {completion[:200]}")
                continue
        except Exception as e:
            print(f"[Consistency] API call error: {e}")
            continue

    return None


def api_batch_consistency_judge(
    predicts: List[str],
    *,
    api_name: Optional[str] = 'Qwen3-VL-235B-A22B-Instruct',
    api_max_workers: int = 32,
    api_kwargs: Optional[Dict] = None,
    client=None,
) -> List[float]:
    """
    批量调用 API 判定推理过程与最终答案的一致性。
    返回与 predicts 等长的 List[float]，每个元素为 0.0 或 1.0。
    """
    import os
    import concurrent.futures as cf
    import traceback
    start_time = time.time()

    n = len(predicts)
    results: List[float] = [0.0] * n

    try:
        max_workers = int(os.environ.get("API_CONSISTENCY_WORKERS", api_max_workers))
    except Exception:
        max_workers = api_max_workers

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i in range(n):
            if _cst_extract_answer(predicts[i]) == '':
                results[i] = 0.0
                continue
            fut = ex.submit(
                _consistency_api_call_wrapper,
                api_name,
                predicts[i],
                client=client,
                api_kwargs=api_kwargs,
            )
            futs.append((i, fut))

        for i, fut in futs:
            try:
                r = fut.result()
                results[i] = r if r is not None else 0.0
            except Exception:
                traceback.print_exc()
                print(f"WARNING: Consistency API judge fail for index {i}, set to 0.0")
                results[i] = 0.0

    end_time = time.time()
    avg_cst = sum(results) / len(results) if results else 0.0
    print(f"[Consistency] Completed {n} samples in {end_time - start_time:.2f}s, avg_consistency={avg_cst:.4f}")
    return results


def rule_then_api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = 'Qwen3-VL-235B-A22B-Instruct-Call',
    api_max_workers: int = 32,
    api_kwargs: Optional[Dict] = None,
    client = None,
    dataset_name: str = "",
    repetition_penalty: bool = False
):
    """
    Now directly delegates to api_batch_judge, which internally uses
    rule_based_match as the fast path, then API for unresolved cases.
    The old two-phase logic (extract_and_check then api_batch_judge)
    was redundant since api_batch_judge already includes rule-based matching.
    """
    return api_batch_judge(
        questions,
        preds,
        gts,
        api_name=api_name,
        api_max_workers=api_max_workers,
        api_kwargs=api_kwargs,
        client=client,
        dataset_name=dataset_name,
        repetition_penalty=repetition_penalty
    )



def compute_score_w_prev_correctness(predicts: List[str], correctness_list: List[float], difficulties: List[str], bootstrap_accs: List[float], format_weight: float = 0.1, tool_weight: float = 0.2, length_penalty_weight = 0.001, resp_lengths = None, ref_resp_lengths = None, consistency_weight: float = 0.5) -> List[Dict[str, float]]:

    """
    bootstrap_accs: List[float], 取值范围 [0.0, 1.0], 代表该题的初始准确率难度
    consistency_weight: 一致性 reward 的权重，默认 0.5
    """
    scores = []

    print(f"current difficulties is {difficulties}")

    # 在函数内部直接调用 API 批量判定一致性
    consistency_list = api_batch_consistency_judge(
        predicts,
        api_name='Qwen3-VL-4B',
        api_max_workers=8,
    )

    for i, (predict, correctness, p) in enumerate(zip(predicts, correctness_list, bootstrap_accs)):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        format_score = format_reward(predict) 
        accuracy_score = 1.0 if correctness == 1.0 else correctness
        is_correct = (accuracy_score == 1.0)

        # 只要出现起始符，就算尝试调用工具
        has_tool_use = "<START_OF_GEN>" in predict

        # ----------------------------------------------------------------
        # Part A: 基于连续函数的工具门控奖励 (Continuous Gating Delta)
        # p = bootstrap_acc (0.0=最难, 1.0=最简单)
        # ----------------------------------------------------------------
        gating_delta = 0.0
        # 防止输入异常值破坏数学函数
        p = max(0.0, min(1.0, float(p))) 

        #### V3
        if has_tool_use:
            if is_correct:
                # 场景1: 有工具且做对 (原 S3 加上工具奖励/惩罚差值)
                # 交点在 p=2/3。困难题重奖，简单题轻微惩罚
                # p=0 -> +0.75, p=2/3 -> +0.167, p=1.0 -> -0.05
                gating_delta = 0.25 + 0.20 * (1.0 - p) + 0.3 * math.cos(math.pi * p)
            else:
                # 场景2: 有工具但做错 (原 S4 加上工具奖励/惩罚差值)
                # 交点在 p=2/3。困难题给探索空间，简单题重罚
                # p=0 -> -0.05, p=2/3 -> -0.30, p=1.0 -> -0.35
                gating_delta = -0.05 - 0.30 * (1.0 - p) + 0.3 * math.cos(math.pi * p)
        else:
            if is_correct:
                # 场景3: 无工具且做对 (独立做对的基础线)
                # 越难的题奖励越高
                # p=0 -> +0.30, p=2/3 -> +0.167, p=1.0 -> +0.10
                gating_delta = 0.10 + 0.20 * (1.0 - p)
            else:
                # 场景4: 无工具且做错 (盲目自信的基础线)
                # 难题重罚，简单题轻罚
                # p=0 -> -0.50, p=2/3 -> -0.30, p=1.0 -> -0.20
                gating_delta = -0.20 - 0.30 * (1.0 - p)
        
        # ----------------------------------------------------------------
        # Part B: 一致性 Reward (只有答案正确时才计入)
        # ----------------------------------------------------------------
        raw_cst = consistency_list[i]
        consistency_score = raw_cst * consistency_weight if is_correct else 0.0

        # ----------------------------------------------------------------
        # Part C: 综合计算
        # ----------------------------------------------------------------
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score + gating_delta + consistency_score

        scores.append(
            {
                "overall": overall_score,
                "filter": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "gating_score": gating_delta,
                "consistency": consistency_score,
            }
        )
    
    # 统计打印部分
    if len(scores) > 0:
          avg_accuracy = sum(s["accuracy"] for s in scores) / len(scores)
          avg_format = sum(s["format"] for s in scores) / len(scores)
          avg_gating = sum(s["gating_score"] for s in scores) / len(scores)
          avg_consistency = sum(s["consistency"] for s in scores) / len(scores)

          tool_call_count = sum(1 for pred in predicts if "<START_OF_GEN>" in pred)
          tool_call_ratio = tool_call_count / len(predicts) if len(predicts) > 0 else 0.0

          # 按难度分桶统计
          diff_tool = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}  # [tool_call_count, total_count]
          diff_acc = {"easy": [], "medium": [], "hard": []}
          diff_gating = {"easy": [], "medium": [], "hard": []}
          diff_cst = {"easy": [], "medium": [], "hard": []}

          for predict, score, diff in zip(predicts, scores, difficulties):
              diff_tool[diff][1] += 1
              if "<START_OF_GEN>" in predict:
                  diff_tool[diff][0] += 1
              diff_acc[diff].append(score["accuracy"])
              diff_gating[diff].append(score["gating_score"])
              diff_cst[diff].append(score["consistency"])

          print("-" * 40)
          print(f"[Reward Summary] Batch Size: {len(scores)}")
          print(f"  - Avg Accuracy Score: {avg_accuracy:.4f}")
          print(f"  - Avg Format Score:   {avg_format:.4f}")
          print(f"  - Avg Gating Delta:   {avg_gating:.4f}")
          print(f"  - Avg Consistency:    {avg_consistency:.4f}")
          print(f"  - Tool Call Count:    {tool_call_count}/{len(predicts)} ({tool_call_ratio:.2%})")
          print(f"  - Last Predict Output (Partial):   {predict[:1000]}...")
          print(f"  --- Per-Difficulty Breakdown ---")
          for d in ("easy", "medium", "hard"):
              tc, total = diff_tool[d]
              ratio = tc / total if total > 0 else 0.0
              avg_a = sum(diff_acc[d]) / len(diff_acc[d]) if diff_acc[d] else 0.0
              avg_g = sum(diff_gating[d]) / len(diff_gating[d]) if diff_gating[d] else 0.0
              avg_c = sum(diff_cst[d]) / len(diff_cst[d]) if diff_cst[d] else 0.0
              print(f"  [{d:>6s}] count={total:4d}, tool_call={tc:4d} ({ratio:.2%}), "
                    f"avg_acc={avg_a:.4f}, avg_gating={avg_g:.4f}, avg_consistency={avg_c:.4f}")
          print("-" * 40)

    return scores