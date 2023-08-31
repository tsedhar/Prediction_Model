from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample feedback data
feedbacks = [
    "I must say, my overall experience of using the application is good.",
    "The accuracy of the performance prediction is closely aligned with the actual result.",
    "The precision of the result could be improved, and a better decision support system provided for the user.",
    "This can achieve through the implementation of additional features influential for the student performance on the virtual learning platform.",
    "The user interface was intuitive and easy to navigate, allowing me to input the student average grades, total clicks, and number of attempts seamlessly.",
    " However, there are a few areas where I believe improvements could enhance the application further.",
    "For instance, Speed and responsiveness could be improved to provide a smoother user experience.",
    "In addition to that, some technical problems should be addressed to ensure consistent performance, though the exception handling techniques were employed throughout the application development.",
    "Finally, I recommend including a wider range of input options such as study behavior, emotions status and study method etc. for future development."

]
# Join all feedbacks into a single string
feedback_text = " ".join(feedbacks)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)

# Plot the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
