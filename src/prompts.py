"""
Prompt strategy library for the prompt sensitivity study.

Tests whether different prompting strategies can close the modality gap
between text-only and image-based math reasoning.

Strategies:
  1. Zero-shot (baseline)       — "Solve this problem. End with #### <answer>"
  2. Zero-shot CoT              — "Think step by step" appended
  3. Instruction priming        — "Read the image very carefully before solving"
  4. Format priming             — "Extract all numbers and relationships first"
  5. Few-shot (1-shot)          — One worked example included in prompt
  6. Few-shot (3-shot)          — Three worked examples included
  7. Self-verification          — "Solve, then verify your answer"
  8. Structured output          — "List: givens, goal, steps, answer"

Each strategy returns (text_prompt, image_prompt) — the prompts to use
for text-only and image-based conditions respectively.
"""

from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  FEW-SHOT EXAMPLES (from GSM8K train set — not in test set)
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "solution": "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = $18 every day at the farmer's market.\n#### 18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "solution": "It takes 2 / 2 = 1 bolt of white fiber.\nSo it takes 2 + 1 = 3 bolts in total.\n#### 3",
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "solution": "The cost of the house and repairs came out to 80,000 + 50,000 = $130,000.\nHe increased the value of the house by 80,000 * 1.5 = $120,000.\nSo the new value is 80,000 + 120,000 = $200,000.\nHis profit is 200,000 - 130,000 = $70,000.\n#### 70000",
    },
]


def _format_examples(n: int) -> str:
    """Format n few-shot examples."""
    parts = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES[:n]):
        parts.append(
            f"Example {i+1}:\n"
            f"Problem: {ex['question']}\n"
            f"Solution: {ex['solution']}\n"
        )
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def zero_shot(question: str) -> Tuple[str, str]:
    """Strategy 1: Basic zero-shot (baseline)."""
    text_prompt = (
        f"Solve the following math problem step by step. "
        f"Show your reasoning, then end with '#### <answer>'.\n\n"
        f"Problem: {question}"
    )
    image_prompt = (
        "The image contains a math word problem. "
        "Read it carefully and solve it step by step. "
        "End with '#### <answer>'."
    )
    return text_prompt, image_prompt


def zero_shot_cot(question: str) -> Tuple[str, str]:
    """Strategy 2: Zero-shot Chain-of-Thought."""
    text_prompt = (
        f"Solve the following math problem. "
        f"Let's think step by step.\n\n"
        f"Problem: {question}\n\n"
        f"Think through each step carefully, then end with '#### <answer>'."
    )
    image_prompt = (
        "The image contains a math word problem. "
        "Let's think step by step. "
        "Read it carefully, work through the logic systematically, "
        "and end with '#### <answer>'."
    )
    return text_prompt, image_prompt


def instruction_priming(question: str) -> Tuple[str, str]:
    """Strategy 3: Explicit instruction to read carefully."""
    text_prompt = (
        f"Read the following problem very carefully. "
        f"Pay close attention to every number, unit, and relationship. "
        f"Then solve it step by step.\n\n"
        f"Problem: {question}\n\n"
        f"End with '#### <answer>'."
    )
    image_prompt = (
        "IMPORTANT: Read the image very carefully before attempting to solve. "
        "Extract every number, unit, and relationship from the text in the image. "
        "Double-check that you've read all values correctly. "
        "Then solve step by step. End with '#### <answer>'."
    )
    return text_prompt, image_prompt


def format_priming(question: str) -> Tuple[str, str]:
    """Strategy 4: Extract information first, then solve."""
    text_prompt = (
        f"Solve the following math problem using this approach:\n"
        f"1. First, list all the numbers and quantities mentioned\n"
        f"2. Identify what is being asked\n"
        f"3. Set up the calculation\n"
        f"4. Compute the answer\n\n"
        f"Problem: {question}\n\n"
        f"End with '#### <answer>'."
    )
    image_prompt = (
        "The image contains a math word problem. Follow these steps:\n"
        "1. First, carefully extract ALL numbers and quantities from the image\n"
        "2. Identify what is being asked\n"
        "3. Set up the calculation\n"
        "4. Compute the answer\n\n"
        "End with '#### <answer>'."
    )
    return text_prompt, image_prompt


def few_shot_1(question: str) -> Tuple[str, str]:
    """Strategy 5: 1-shot with worked example."""
    examples = _format_examples(1)
    text_prompt = (
        f"Here is a worked example of solving a math problem:\n\n"
        f"{examples}\n"
        f"Now solve this problem in the same way:\n\n"
        f"Problem: {question}\n\n"
        f"End with '#### <answer>'."
    )
    image_prompt = (
        f"Here is a worked example of solving a math problem:\n\n"
        f"{examples}\n"
        f"The image contains a new math problem. "
        f"Solve it in the same way. End with '#### <answer>'."
    )
    return text_prompt, image_prompt


def few_shot_3(question: str) -> Tuple[str, str]:
    """Strategy 6: 3-shot with worked examples."""
    examples = _format_examples(3)
    text_prompt = (
        f"Here are worked examples of solving math problems:\n\n"
        f"{examples}\n"
        f"Now solve this problem in the same way:\n\n"
        f"Problem: {question}\n\n"
        f"End with '#### <answer>'."
    )
    image_prompt = (
        f"Here are worked examples of solving math problems:\n\n"
        f"{examples}\n"
        f"The image contains a new math problem. "
        f"Solve it in the same way. End with '#### <answer>'."
    )
    return text_prompt, image_prompt


def self_verify(question: str) -> Tuple[str, str]:
    """Strategy 7: Solve then verify."""
    text_prompt = (
        f"Solve the following math problem step by step. "
        f"After finding your answer, verify it by plugging it back in "
        f"or checking your arithmetic. If you find an error, correct it.\n\n"
        f"Problem: {question}\n\n"
        f"End with '#### <answer>'."
    )
    image_prompt = (
        "The image contains a math word problem. "
        "Solve it step by step. After finding your answer, "
        "verify it by checking your arithmetic. "
        "If you find an error, correct it. "
        "End with '#### <answer>'."
    )
    return text_prompt, image_prompt


def structured_output(question: str) -> Tuple[str, str]:
    """Strategy 8: Force structured reasoning."""
    text_prompt = (
        f"Solve the following math problem using this exact format:\n\n"
        f"GIVEN: [list all numbers and facts]\n"
        f"FIND: [what we need to calculate]\n"
        f"STEPS:\n"
        f"  1. [first calculation]\n"
        f"  2. [next calculation]\n"
        f"  ...\n"
        f"ANSWER: [final number]\n\n"
        f"Problem: {question}\n\n"
        f"End with '#### <answer>'."
    )
    image_prompt = (
        "The image contains a math word problem. Solve it using this exact format:\n\n"
        "GIVEN: [list all numbers and facts from the image]\n"
        "FIND: [what we need to calculate]\n"
        "STEPS:\n"
        "  1. [first calculation]\n"
        "  2. [next calculation]\n"
        "  ...\n"
        "ANSWER: [final number]\n\n"
        "End with '#### <answer>'."
    )
    return text_prompt, image_prompt


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_STRATEGIES = {
    "zero_shot": {
        "fn": zero_shot,
        "description": "Basic zero-shot (baseline)",
        "category": "baseline",
    },
    "zero_shot_cot": {
        "fn": zero_shot_cot,
        "description": "Zero-shot Chain-of-Thought ('let's think step by step')",
        "category": "cot",
    },
    "instruction_priming": {
        "fn": instruction_priming,
        "description": "Explicit instruction to read image carefully",
        "category": "priming",
    },
    "format_priming": {
        "fn": format_priming,
        "description": "Extract information first, then solve",
        "category": "priming",
    },
    "few_shot_1": {
        "fn": few_shot_1,
        "description": "1-shot with worked example from GSM8K train",
        "category": "few_shot",
    },
    "few_shot_3": {
        "fn": few_shot_3,
        "description": "3-shot with worked examples from GSM8K train",
        "category": "few_shot",
    },
    "self_verify": {
        "fn": self_verify,
        "description": "Solve then verify/check arithmetic",
        "category": "verification",
    },
    "structured_output": {
        "fn": structured_output,
        "description": "Forced structured reasoning (GIVEN/FIND/STEPS/ANSWER)",
        "category": "structured",
    },
}


def get_prompts(strategy_name: str, question: str) -> Tuple[str, str]:
    """Get (text_prompt, image_prompt) for a given strategy."""
    if strategy_name not in PROMPT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(PROMPT_STRATEGIES.keys())}"
        )
    return PROMPT_STRATEGIES[strategy_name]["fn"](question)


def list_strategies():
    """Print all available prompt strategies."""
    print("\nPrompt Strategies:")
    print("-" * 70)
    for name, info in PROMPT_STRATEGIES.items():
        print(f"  {name:25s}  [{info['category']:12s}]  {info['description']}")
    print()
