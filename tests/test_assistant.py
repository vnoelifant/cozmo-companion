import pytest

from src.cozmo_companion.assistant import is_feedback_inquiry_present


@pytest.mark.parametrize(
    "bot_text, expected",
    [
        pytest.param(
            "North America is a continent. Did this answer help you?",
            True,
            id="Question 1",
        ),
        pytest.param(
            "South America is a continent. Did this answer surprise you?",
            True,
            id="Question 2",
        ),
        pytest.param(
            "Soccer is the most popular sports in Germany.", False, id="No question"
        ),
    ],
)
def test_is_feedback_inquiry_present(bot_text, expected):
    feedback_present = is_feedback_inquiry_present(bot_text)

    assert feedback_present == expected
