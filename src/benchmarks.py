"""
Unified benchmark loader for multiple math and reasoning datasets.

All datasets are FREE and loaded from HuggingFace Hub.

Supported benchmarks:
  ── Math (text-based, can be rendered as images) ──────────────
  1. GSM8K      — 1,319 grade-school math word problems
  2. SVAMP      — 1,000 math word problems (variation-resistant)
  3. AQuA-RAT   — 254 algebraic word problems (multiple choice)
  4. MATH       — 5,000 competition-level math (LaTeX heavy)

  ── Visual Math (natively multimodal, come with images) ───────
  5. MathVista  — 6,141 visual math reasoning problems

  ── Non-Math Visual Reasoning ─────────────────────────────────
  6. ScienceQA  — 21,208 multimodal science questions
  7. AI2D       — 4,903 science diagram questions
  8. ChartQA    — 2,500 chart understanding questions

Each benchmark returns a list of BenchmarkItem with unified fields.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from PIL import Image


@dataclass
class BenchmarkItem:
    """Unified format for a single benchmark problem."""
    id: int
    question: str
    reference_answer: str          # canonical answer string
    reference_number: Optional[float] = None  # numeric answer if applicable
    choices: Optional[List[str]] = None       # for multiple-choice
    correct_choice: Optional[int] = None      # index into choices
    image: Optional[Image.Image] = None       # native image (for visual benchmarks)
    category: str = ""                        # sub-category within benchmark
    difficulty: str = ""                      # difficulty level if available
    metadata: dict = field(default_factory=dict)


def extract_number_from_answer(text: str) -> Optional[float]:
    """Extract numeric value from various answer formats."""
    if not text:
        return None
    # GSM8K: #### <number>
    canon = re.search(r"####\s*([\-\d,\.]+)", text)
    if canon:
        try:
            return float(canon.group(1).replace(",", ""))
        except ValueError:
            pass
    # Try last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    numbers = [n for n in numbers if n.replace(",", "").replace(".", "").replace("-", "")]
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_gsm8k(num_problems=None) -> List[BenchmarkItem]:
    """GSM8K — grade-school math word problems."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        ref_num = extract_number_from_answer(row["answer"])
        items.append(BenchmarkItem(
            id=i,
            question=row["question"],
            reference_answer=row["answer"],
            reference_number=ref_num,
            category="grade_school_math",
        ))
    print(f"GSM8K: loaded {len(items)} problems")
    return items


def load_svamp(num_problems=None) -> List[BenchmarkItem]:
    """SVAMP — simple variations on arithmetic math problems."""
    ds = load_dataset("ChilleD/SVAMP", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # SVAMP has 'Body' + 'Question' fields and 'Answer' as a number
        question = f"{row['Body']} {row['Question']}"
        answer = str(row["Answer"])
        items.append(BenchmarkItem(
            id=i,
            question=question,
            reference_answer=answer,
            reference_number=float(row["Answer"]),
            category=row.get("Type", "arithmetic"),
        ))
    print(f"SVAMP: loaded {len(items)} problems")
    return items


def load_aqua_rat(num_problems=None) -> List[BenchmarkItem]:
    """AQuA-RAT — algebraic word problems (multiple choice)."""
    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # Format choices into the question
        options = row["options"]
        choices_text = "\n".join(options)
        full_question = f"{row['question']}\n\nOptions:\n{choices_text}"

        # correct answer is a letter (A-E)
        correct_letter = row["correct"]
        correct_idx = ord(correct_letter) - ord("A")

        items.append(BenchmarkItem(
            id=i,
            question=full_question,
            reference_answer=correct_letter,
            choices=options,
            correct_choice=correct_idx,
            category="algebra",
        ))
    print(f"AQuA-RAT: loaded {len(items)} problems")
    return items


def load_math_dataset(num_problems=None) -> List[BenchmarkItem]:
    """MATH — competition-level math problems."""
    ds = load_dataset("hendrycks/competition_math", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # MATH answers are in \boxed{...} format
        answer = row["solution"]
        # Extract boxed answer
        boxed = re.search(r"\\boxed\{([^}]+)\}", answer)
        ref_answer = boxed.group(1) if boxed else answer
        ref_num = extract_number_from_answer(ref_answer)

        items.append(BenchmarkItem(
            id=i,
            question=row["problem"],
            reference_answer=ref_answer,
            reference_number=ref_num,
            category=row.get("type", "math"),
            difficulty=str(row.get("level", "")),
        ))
    print(f"MATH: loaded {len(items)} problems")
    return items


def load_mathvista(num_problems=None) -> List[BenchmarkItem]:
    """
    MathVista — visual math reasoning (natively multimodal).
    These come with actual images — no need to render text as images.
    """
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        question = row.get("question", row.get("query", ""))
        answer = str(row.get("answer", ""))
        ref_num = extract_number_from_answer(answer)

        # MathVista includes images
        image = row.get("decoded_image", row.get("image", None))
        if image is not None and not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                image = None

        choices = row.get("choices", None)

        items.append(BenchmarkItem(
            id=i,
            question=question,
            reference_answer=answer,
            reference_number=ref_num,
            image=image,
            choices=choices,
            category=row.get("metadata", {}).get("category", "visual_math") if isinstance(row.get("metadata"), dict) else "visual_math",
        ))
    print(f"MathVista: loaded {len(items)} problems")
    return items


def load_scienceqa(num_problems=None) -> List[BenchmarkItem]:
    """ScienceQA — multimodal science questions (multiple choice)."""
    ds = load_dataset("derek-thomas/ScienceQA", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # Build question with choices
        question = row.get("question", "")
        choices = row.get("choices", [])
        if choices:
            choices_text = "\n".join([f"{chr(65+j)}) {c}" for j, c in enumerate(choices)])
            full_question = f"{question}\n\n{choices_text}"
        else:
            full_question = question

        answer_idx = row.get("answer", 0)
        ref_answer = chr(65 + answer_idx) if isinstance(answer_idx, int) else str(answer_idx)

        # ScienceQA has optional images
        image = row.get("image", None)
        if image is not None and not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                image = None

        items.append(BenchmarkItem(
            id=i,
            question=full_question,
            reference_answer=ref_answer,
            choices=choices,
            correct_choice=answer_idx if isinstance(answer_idx, int) else None,
            image=image,
            category=row.get("subject", "science"),
        ))
    print(f"ScienceQA: loaded {len(items)} problems")
    return items


def load_ai2d(num_problems=None) -> List[BenchmarkItem]:
    """AI2D — science diagram understanding (multiple choice)."""
    ds = load_dataset("lmms-lab/ai2d", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        question = row.get("question", "")
        choices = row.get("options", row.get("choices", []))

        if isinstance(choices, list) and choices:
            choices_text = "\n".join([f"{chr(65+j)}) {c}" for j, c in enumerate(choices)])
            full_question = f"{question}\n\n{choices_text}"
        else:
            full_question = question

        answer = row.get("answer", 0)
        if isinstance(answer, int):
            ref_answer = chr(65 + answer)
            correct_idx = answer
        else:
            ref_answer = str(answer)
            correct_idx = None

        image = row.get("image", None)
        if image is not None and not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                image = None

        items.append(BenchmarkItem(
            id=i,
            question=full_question,
            reference_answer=ref_answer,
            choices=choices if isinstance(choices, list) else None,
            correct_choice=correct_idx,
            image=image,
            category="science_diagram",
        ))
    print(f"AI2D: loaded {len(items)} problems")
    return items


def load_chartqa(num_problems=None) -> List[BenchmarkItem]:
    """ChartQA — chart understanding and reasoning."""
    ds = load_dataset("lmms-lab/ChartQA", split="test")
    if num_problems:
        ds = ds.select(range(min(num_problems, len(ds))))

    items = []
    for i, row in enumerate(ds):
        question = row.get("question", row.get("query", ""))
        answer = str(row.get("answer", row.get("label", "")))
        ref_num = extract_number_from_answer(answer)

        image = row.get("image", None)
        if image is not None and not isinstance(image, Image.Image):
            try:
                image = Image.open(image).convert("RGB")
            except Exception:
                image = None

        items.append(BenchmarkItem(
            id=i,
            question=question,
            reference_answer=answer,
            reference_number=ref_num,
            image=image,
            category="chart_understanding",
        ))
    print(f"ChartQA: loaded {len(items)} problems")
    return items


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY & FACTORY
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_REGISTRY = {
    # Text-based math (render as images for Condition 2)
    "gsm8k":    {"loader": load_gsm8k,     "type": "text_math",   "has_images": False},
    "svamp":    {"loader": load_svamp,      "type": "text_math",   "has_images": False},
    "aqua_rat": {"loader": load_aqua_rat,   "type": "text_math",   "has_images": False},
    "math":     {"loader": load_math_dataset, "type": "text_math", "has_images": False},

    # Visual math (native images)
    "mathvista": {"loader": load_mathvista, "type": "visual_math", "has_images": True},

    # Non-math visual reasoning (native images)
    "scienceqa": {"loader": load_scienceqa, "type": "visual_reasoning", "has_images": True},
    "ai2d":      {"loader": load_ai2d,      "type": "visual_reasoning", "has_images": True},
    "chartqa":   {"loader": load_chartqa,    "type": "visual_reasoning", "has_images": True},
}


def load_benchmark(name: str, num_problems=None) -> List[BenchmarkItem]:
    """Load a benchmark by name."""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: {name}. "
            f"Available: {list(BENCHMARK_REGISTRY.keys())}"
        )
    return BENCHMARK_REGISTRY[name]["loader"](num_problems)


def get_benchmark_info(name: str) -> dict:
    """Get metadata about a benchmark."""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}")
    return BENCHMARK_REGISTRY[name]


def list_benchmarks():
    """Print all available benchmarks."""
    print("\nAvailable Benchmarks:")
    print("-" * 70)
    for name, info in BENCHMARK_REGISTRY.items():
        print(f"  {name:12s}  type={info['type']:20s}  has_images={info['has_images']}")
    print()
