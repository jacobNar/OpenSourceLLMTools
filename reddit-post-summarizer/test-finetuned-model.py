# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification",
                model="jacobNar/bert-reddit-classifier-v6", device="cuda")

examples = [
    "8 months ago I launched this database that analyzes real user problems from G2 reviews, App Store feedback, and Reddit threads to help founders find their next profitable SaaS idea. It went on to make $21,000 which is kinda insane for me to think about.I was very new to this whole marketing thing when launching it (because I failed 8 times, with all of them getting <50 users and 0 dollars from a product hunt launch lmao). The main channels I picked for this product where I would change it all were X, Reddit, and some founder communities (discord/slack). So I just started building a following together with my app as it grew. This is what every founder should imo as long as you build a good or at least interesting product (for an audience. dw no one is going to steal your idea if you start posting if you are done 50% of your mvp). As my product grew so did my following. It was like a self-feeding cycle.Here are my stats so far:1,700+ total signups66 active paying subscribers$21,000 total revenue$5,300 in just the last 2 months24,000+ unique website visitors in the past 2 months ",
    "Title says it all. See a lot of bs posts across reddit about 10x this or went to 100k MRR in a month that. So I dropped a link with a screenshot for proof on how we've built our content engine and 7x'd our impressions and scaled our LLM visibility in under 90 days. Drop your company info and I can share some strategy for how you could scale as well"
]

result = pipe(examples[1])

print(result)
