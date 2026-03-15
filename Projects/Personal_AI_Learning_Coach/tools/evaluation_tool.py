from typing import List
from agno.tools import tool

@tool
def evaluate_quiz(user_answers: List[str], correct_answers: List[str]):
    """
    Compare user answers with correct answers and return score.
    """

    total = len(correct_answers)
    correct = 0

    for u, c in zip(user_answers, correct_answers):
        if u.strip().lower() == c.strip().lower():
            correct += 1

    score = (correct / total) * 100

    result = {
        "score": score,
        "correct": correct,
        "total": total
    }

    if score < 70 and score > 40:
        result["status"] = "Average"
    elif score > 70:
        result["status"] = "Good Performance"
    else:
        result["status"] = "Bad Performance"

    return result

