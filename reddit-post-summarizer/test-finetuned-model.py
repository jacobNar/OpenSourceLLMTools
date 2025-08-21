# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification",
                model="jacobNar/bert-reddit-classifier-v2")

example = "Hi everyone! I'm excited to share that Flag Quiz Master is now live on App Store! If you love geography, trivia, or just want to learn more about the world's flags, this app is for you. I built it as a fun way to test and improve your knowledge of country flags from around the globe. App features: - Multiple game modes (Classic, Time Attack, Daily Challenges) - Different difficulty levels to match your skill - Track your progress and achievements (work in progress lol) - Learn flags from 195+ countries It's 100% free! Here's the app link: Flag Quiz Master on App Store Would really appreciate if you could check it out and leave some feedback! As an indie developer, every download and review helps tremendously "

result = pipe(example)

print(result)
