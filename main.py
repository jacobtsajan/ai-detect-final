from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize the message variable at the beginning of the function.
    # This ensures it's always defined, even for a GET request.
    message = " "  # Or any initial message you'd like to display on first load.
    
    if request.method == 'POST':
        # Get the message from the form.
        input_message = request.form['message']
        
        # Load the pre-trained models.
        # It's better practice to load these once outside the function,
        # but for this example, we'll keep it here as per your original code.
        try:
            clf_svm = pickle.load(open('clf.pkl','rb'))
            tfidf = pickle.load(open('tfidf.pkl','rb'))
            
            # Transform the input text using the loaded TF-IDF vectorizer.
            text = tfidf.transform([input_message])
            
            # Predict the result.
            result = clf_svm.predict(text)
            
            # Update the message based on the prediction.
            if(result == 1):
                message = 'The text is likely written by AI'
            else:
                message = "The text is likely written by Human"
        except FileNotFoundError:
            message = "Error: Model files (clf.pkl or tfidf.pkl) not found."

    # The message variable is now guaranteed to have a value.
    return render_template('main.html', params = message)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
