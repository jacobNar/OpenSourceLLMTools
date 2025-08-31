# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification",
                model="jacobNar/distilbert-5batch-3epoch-reddit-v3", device="cuda")

examples = [
    "Reddit user is seeking information on a product that combines a kitchen scale with dishware, allowing users to track calories consumed from meals without having to weigh food multiple times. The desired product would be a plastic bowl-shaped shell with a built-in scale and display screen or app connectivity. This concept aims to simplify meal tracking for individuals monitoring their calorie intake for health reasons.",
    "A college student and solo founder has launched Revast, an AI-powered study app that transforms PDFs, PPTs, and YouTube lectures into notes and quizzes. The creator is seeking feedback on features, UX/UI tips, and marketing strategies to help grow the app among students, with a focus on simplicity and user engagement.",
    "A Reddit user, JoeyJoey1022, has created a free iOS app called Catspace that uses AI to identify cat breeds based on photos. The app aims to provide fun tools for cat lovers to learn about their pets and share them with others, with optional paid subscription features. The developer is seeking honest feedback from users to improve the app's accuracy and features.",
    "A user in a Perfect Hotel app game is experiencing an issue where some rooms don't get cleaned even after fully upgrading the maid, suggesting a potential bug or design flaw. The user has checked their progress and resources to ensure they have enough credits and upgrades to maintain cleanliness. This issue seems to be inconsistent across different rooms and floors.",
    "A developer has released an app called Mindcast, which uses voice journaling to help users understand themselves better. The app transcribes and analyzes spoken thoughts, providing features such as mood analysis, organization, and tracking progress. For a limited time, the Pro version is available for free, with plans to add additional languages in the future.",
    "Im looking for something that works like the OG iTunes. I would like to organize my music on my computer and then sync my music to my phone.",
    "Im looking to create community and wonder what is better element, discord or what up app?",
    "I'm looking for a completely free software that could capture live audio from a webpage and translate it in real time into text. Nothing fancy, simple GUI and easy to use (I don't want to code anything, you should click on an exe and that's it). It should run in Windows, it has to be software for a desktop PC, not a cell phone app.",
    "Hi everyone, I have recently launched my product and got a few users. Now, I want to inform them about new features, but I am new to marketing and have no Idea how this actually works. So, what tools do you use to send product update emails? How often do you send them? What do I need to keep in mind to get the best out of it?"
]

result = pipe(examples[8])
# result2 = pipe(examples[1])
print(result)
# print(result2)
