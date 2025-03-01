from fsrs import Scheduler, Card, Rating, ReviewLog
from datetime import timedelta, datetime, timezone

scheduler = Scheduler(
    parameters = (0.4026, 1.1839, 3.1730, 15.6910, 7.1949, 0.5345, 1.4604, 0.0046, 1.5458, 0.1192, 1.0193, 1.9395, 0.1100, 0.2961, 2.2698, 0.2315, 2.9898, 0.5166, 0.6621),
    desired_retention = 0.9,
    learning_steps = (timedelta(minutes=1), timedelta(minutes=10)),
    relearning_steps = (timedelta(minutes=10),),
    maximum_interval = 36500,
    enable_fuzzing = True
)

card = Card()

def reviewCard(rating, card):
    card, review_log = scheduler.review_card(card, rating)
    return card, review_log

while True:
    rating = int(input().strip())
    card, review_log = reviewCard(rating, card)
    print(card)

# Rating.Again (==1) forgot the card
# Rating.Hard (==2) remembered the card with serious difficulty
# Rating.Good (==3) remembered the card after a hesitation
# Rating.Easy (==4) remembered the card easily
