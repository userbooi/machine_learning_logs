"""
Extractive question answering with a Hugging Face model.

Given a (question, context) pair, the model finds the span of text inside the
context that best answers the question. It does not generate new text; the
answer is always a substring of the context.

How it works:
  1. The question and context are tokenized together as one sequence:
         [CLS] question tokens [SEP] context tokens [SEP]
  2. The model outputs two scores (logits) for every token:
         start_logits -> how likely this token is the START of the answer
         end_logits   -> how likely this token is the END of the answer
  3. We pick the (start, end) pair with the highest combined score, where
     both tokens are inside the context and start <= end.
  4. Token offsets map that span back to the original context characters.

Model: distilbert/distilbert-base-cased-distilled-squad
  - DistilBERT fine-tuned on SQuAD v1.1 (~260 MB, runs fine on CPU)

Install:
    pip install transformers torch

Run:
    python question_answering.py                # runs the built-in examples
    python question_answering.py --interactive  # type your own context/questions
"""

import argparse

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"
MAX_ANSWER_TOKENS = 30

EXAMPLES = [
    {
        "context": (
            "The Amazon rainforest covers much of the Amazon basin of South America. "
            "This basin encompasses 7,000,000 square kilometres, of which 5,500,000 "
            "square kilometres are covered by the rainforest. The majority of the forest "
            "is contained within Brazil, with 60% of the rainforest, followed by Peru "
            "with 13%, and Colombia with 10%."
        ),
        "questions": [
            "How large is the Amazon basin?",
            "Which country contains most of the rainforest?",
            "What percentage of the rainforest is in Peru?",
        ],
    },
    {
        "context": (
            "Python is a high-level programming language created by Guido van Rossum "
            "and first released in 1991. Its design philosophy emphasizes code "
            "readability with the use of significant indentation."
        ),
        "questions": [
            "Who created Python?",
            "When was Python first released?",
            "What does Python's design philosophy emphasize?",
        ],
    },
]


def answer_question(tokenizer, model, question, context):
    """Return (answer_text, confidence) for a question about the given context."""
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",  # only trim the context, never the question
        return_offsets_mapping=True,
    )
    offsets = inputs.pop("offset_mapping")[0]
    # sequence_ids marks each token: 0 = question, 1 = context, None = special token
    sequence_ids = inputs.sequence_ids(0)

    with torch.no_grad():
        outputs = model(**inputs)

    start_probs = torch.softmax(outputs.start_logits[0], dim=-1)
    end_probs = torch.softmax(outputs.end_logits[0], dim=-1)

    context_idx = [i for i, s in enumerate(sequence_ids) if s == 1]
    best_score, best_span = 0.0, None
    for start in context_idx:
        for end in context_idx:
            if end < start or end - start + 1 > MAX_ANSWER_TOKENS:
                continue
            score = float(start_probs[start] * end_probs[end])
            if score > best_score:
                best_score, best_span = score, (start, end)

    if best_span is None:
        return "", 0.0

    start_char = int(offsets[best_span[0]][0])
    end_char = int(offsets[best_span[1]][1])
    return context[start_char:end_char], best_score


def run_examples(tokenizer, model):
    for i, example in enumerate(EXAMPLES, start=1):
        print(f"\n=== Example {i} ===")
        print(f"Context: {example['context']}\n")
        for question in example["questions"]:
            answer, score = answer_question(tokenizer, model, question, example["context"])
            print(f"Q: {question}")
            print(f"A: {answer}  [confidence {score:.2%}]\n")


def run_interactive(tokenizer, model):
    print("Paste a context paragraph, then ask questions about it.")
    context = input("\nContext: ").strip()
    print("Ask questions (blank line to quit, 'new' for a new context).")
    while True:
        question = input("\nQ: ").strip()
        if not question:
            break
        if question.lower() == "new":
            context = input("\nContext: ").strip()
            continue
        answer, score = answer_question(tokenizer, model, question, context)
        print(f"A: {answer}  [confidence {score:.2%}]")


def main():
    parser = argparse.ArgumentParser(description="Hugging Face question answering demo")
    parser.add_argument("--interactive", action="store_true", help="enter your own context and questions")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()

    if args.interactive:
        run_interactive(tokenizer, model)
    else:
        run_examples(tokenizer, model)


if __name__ == "__main__":
    main()
