import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import sqlite3

sqlite3.register_adapter(datetime, lambda x: x.isoformat())  # Store as ISO format
sqlite3.register_converter("DATETIME", lambda x: datetime.fromisoformat(x.decode()))

app = Flask(__name__)
CORS(app)

# Load medical datasets
class MedicalKnowledgeBase:
    def __init__(self):
        # Load datasets
        self.load_datasets()
        # Initialize models
        self.init_models()
        # Create database
        self.init_database()

    def load_datasets(self):
        try:
            self.medical_data = {
                "sleep_health": [
                    "Maintain a consistent sleep schedule",
                    "Aim for 7-9 hours of sleep per night",
                    "Create a relaxing bedtime routine",
                    "Avoid screens before bedtime",
                    "Keep your bedroom cool and dark",
                    "Limit caffeine intake after noon"
                ],
                "nutrition": [
                    "Eat a balanced diet with plenty of fruits and vegetables",
                    "Stay hydrated by drinking water throughout the day",
                    "Limit processed foods and added sugars",
                    "Include lean proteins in your diet",
                    "Choose whole grains over refined grains"
                ],
                "What is your name?": [
                    "My name is Florence AI, am an Advanced health-care bot. Feel free to ask me any health related question."
                ],
                "What are the symptoms of diabetes?": 
                ["Common symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision."],
    "How can I lower my blood pressure?":
      ["Reduce salt intake, exercise regularly, manage stress, and eat a balanced diet."],
    "What is the normal body temperature?":
      ["The normal body temperature is around 98.6°F (37°C)."],
    "How do I treat a fever at home?":
      ["Stay hydrated, rest, and take fever reducers like paracetamol."],
    "What are the early signs of COVID-19?":
      ["Early signs include fever, cough, fatigue, and loss of taste or smell."],
    "How can I improve my immune system?":
      ["Eat a balanced diet, get enough sleep, exercise regularly, and reduce stress."],
    "What are the symptoms of a heart attack?":
      ["Chest pain, shortness of breath, nausea, cold sweat, and discomfort in the arms or jaw."],
    "What should I eat for a healthy heart?":
      ["Consume fruits, vegetables, whole grains, lean proteins, and avoid saturated fats."],
    "What are the common causes of headaches?":
      ["Stress, dehydration, lack of sleep, eye strain, and sinus infections."],
    "How can I manage stress effectively?":
      ["Practice meditation, deep breathing, exercise, and maintain a healthy work-life balance."],
    "What are the symptoms of food poisoning?":
      ["Nausea, vomiting, diarrhea, stomach cramps, and fever."],
    "How can I prevent dehydration?":
      ["Drink enough water daily, especially in hot weather or after physical activity."],
    "What are the signs of depression?":
      ["Persistent sadness, loss of interest, fatigue, changes in appetite, and sleep disturbances."],
    "How can I get better sleep?":
      ["Maintain a sleep schedule, limit screen time before bed, and avoid caffeine in the evening."],
    "What are the symptoms of iron deficiency anemia?":
      ["Fatigue, pale skin, shortness of breath, dizziness, and cold hands or feet."],
    "How can I maintain good eye health?":
      ["Eat vitamin-rich foods, reduce screen time, and wear UV-protected sunglasses."],
    "What are the symptoms of an allergic reaction?":
      ["Eat vitamin-rich foods, reduce screen time, and wear UV-protected sunglasses."],
    "How can I relieve back pain?":
      ["Maintain good posture, stretch regularly, and apply heat or ice packs."],
    "What is the best way to lose weight healthily?":
      ["Eat a balanced diet, exercise regularly, and avoid processed foods."],
    "How can I boost my energy levels naturally?":
      ["Stay hydrated, eat a nutritious diet, exercise, and get enough sleep."]
            }

            # Load medical terminology and definitions
            # self.medical_terms = pd.read_csv('https://raw.githubusercontent.com/OpenI18N/medical-terms/master/medical-terms.csv')
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            # Fallback to basic dataset if loading fails
            self.medical_data = {}
            # self.medical_terms = pd.DataFrame()

    def init_models(self):
        try:
            # Initialize sentence transformer for semantic search
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            # Initialize medical intent classifier
            self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
            self.model = AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1')
            
        except Exception as e:
            print(f"Error initializing models: {e}")

    def init_database(self):
        # Initialize SQLite database for conversation history
        self.conn = sqlite3.connect('florence.db', check_same_thread=False,  detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                assistant_response TEXT,
                timestamp DATETIME
            )
        ''')
        self.conn.commit()

class FlorenceAI:
    def __init__(self):
        self.kb = MedicalKnowledgeBase()
        
    def process_message(self, message):
        try:
            # Encode user message
            message_embedding = self.kb.sentence_model.encode(message)
            
            # Find most relevant category
            best_category = None
            best_similarity = -1
            
            for category, responses in self.kb.medical_data.items():
                category_embedding = self.kb.sentence_model.encode(category)
                similarity = cosine_similarity(
                    [message_embedding],
                    [category_embedding]
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category
            
            # Generate response based on category
            if best_category and best_similarity > 0.3:
                responses = self.kb.medical_data[best_category]
                response = self.generate_response(message, responses, best_category)
            else:
                response = self.generate_fallback_response()
            
            # Store conversation in database
            self.store_conversation(message, response)
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."

    def generate_response(self, message, responses, category):
        try:
            # Generate contextual response
            intro = f"Alright! Regarding {category.replace('_', ' ')}, here are some recommendations:"
            
            # Select relevant responses
            selected_responses = responses[:3]  # Limit to 3 recommendations
            
            # Format response
            response = f"{intro}\n\n"
            for i, rec in enumerate(selected_responses, 1):
                response += f"{i}. {rec}\n"
            
            response += "\nWould you like more specific information about any health topic !"
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self.generate_fallback_response()

    def generate_fallback_response(self):
        return ("I apologize, but I'm not able to provide specific advice about that topic. "
                "Please consult with a healthcare professional for personalized medical advice.")

    def store_conversation(self, message, response):
        try:
            self.kb.cursor.execute('''
                INSERT INTO conversations (user_message, assistant_response, timestamp)
                VALUES (?, ?, ?)
            ''', (message, response, datetime.now()))
            self.kb.conn.commit()
        except Exception as e:
            print(f"Error storing conversation: {e}")

# Initialize Florence
florence = FlorenceAI()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        response = florence.process_message(message)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)