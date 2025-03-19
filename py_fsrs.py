from fsrs import Scheduler, Card, Rating, ReviewLog
from datetime import timedelta, datetime, timezone
import json

def initializeFSRS() :
    scheduler = Scheduler(
        parameters = (0.4026, 1.1839, 3.1730, 15.6910, 7.1949, 0.5345, 1.4604, 0.0046, 1.5458, 0.1192, 1.0193, 1.9395, 0.1100, 0.2961, 2.2698, 0.2315, 2.9898, 0.5166, 0.6621),
        desired_retention = 0.9,
        learning_steps = (timedelta(minutes=1), timedelta(minutes=10)),
        relearning_steps = (timedelta(minutes=10),),
        maximum_interval = 36500,
        enable_fuzzing = True
    )
    return scheduler

def reviewCard(scheduler, card, rating):
    card, review_log = scheduler.review_card(card, rating)
    return card, review_log

def getNextReviewInterval(scheduler, card, rating):
    card, _ =  reviewCard(scheduler, card, rating)
    interval_seconds = (card.due - card.last_review).total_seconds()
        
    if interval_seconds < 3600:
        interval = f"{round(interval_seconds / 60)} minutes"
    elif interval_seconds < 86400:
        interval = f"{round(interval_seconds / 3600)} hours"
    else:
        interval = f"{round(interval_seconds / 86400)} days"

    return interval

scheduler = initializeFSRS()
