# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification",
                model="jacobNar/bert-reddit-classifier-v3")

example = "On a recent flight, my husband and I booked E+ seats (aisle for him, window for me). As we were approaching our seating row, a very young child was sitting in the middle seat and a woman (on the opposite side of the plane in that same row) sitting in the middle seat was discussing a “seat swap” with a passenger standing in the aisle of that row. Well, it turns out the passenger she was talking with was assigned the window seat next to her. She had asked if he wouldn’t mind switching seats so that her daughter (the young child that was seated in the middle on the opposite side) could be seated next to her. (It was a completely full flight so I’m assuming they must’ve been standby passengers and is possibly why they were not seated together.) This guy was kind in that he agreed to let the child take his window seat. "

result = pipe(example)

print(result)
