# TASK3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset
data = {
    "title": [
        "AI internship guide for beginners",
        "How to get data science internship",
        "Web development internship roadmap",
        "Machine learning internship preparation",
        "Top skills for software internships",
        "Python projects for internship",
        "How to crack internship interviews",
        "Resume tips for internships"
    ]
}

df = pd.DataFrame(data)

# Recommendation function
def recommend(user_input):
    # Combine dataset + user input
    all_titles = df["title"].tolist()
    all_titles.append(user_input)

    # Convert text to vectors
    cv = CountVectorizer()
    matrix = cv.fit_transform(all_titles)

    # Similarity
    similarity = cosine_similarity(matrix)

    # Last index = user input
    scores = list(enumerate(similarity[-1]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\n🎯 Recommended Videos:\n")
    for i in scores[1:4]:
        print(all_titles[i[0]])

# User input
user_input = input("Enter topic: ")
recommend(user_input)
