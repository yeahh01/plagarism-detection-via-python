from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Two sample documents
document1 = """
Python is a popular programming language.
It is used for web development, data science,
artificial intelligence and automation.
"""

document2 = """
Python is a widely used programming language.
It is useful for data science, web development,
artificial intelligence and automation.
"""


# Convert the documents into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([document1, document2])


# Calculate cosine similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])


# Convert the result into a percentage
similarity_percentage = similarity[0][0] * 100


# Display the result
print("=================================")
print("       PLAGIARISM DETECTOR")
print("=================================")

print(f"Similarity: {similarity_percentage:.2f}%")

if similarity_percentage >= 70:
    print("Result: High similarity - possible plagiarism")
elif similarity_percentage >= 40:
    print("Result: Moderate similarity")
else:
    print("Result: Low similarity")