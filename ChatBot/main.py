import random
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# First, let's add better error handling and debugging for API keys
def check_api_keys():
    weather_key = os.getenv('WEATHER_API_KEY')
    news_key = os.getenv('NEWS_API_KEY')
    
    if not weather_key:
        print("\n⚠️  Warning: Weather API key not found in .env file")
        print("Please add WEATHER_API_KEY=your_key_here to your .env file")
    elif len(weather_key) < 20:  # Basic validation
        print("\n⚠️  Warning: Weather API key appears invalid")
        print("Please check your WEATHER_API_KEY in .env file")
        
    if not news_key:
        print("\n⚠️  Warning: News API key not found in .env file")
        print("Please add NEWS_API_KEY=your_key_here to your .env file")
    elif len(news_key) < 20:  # Basic validation
        print("\n⚠️  Warning: News API key appears invalid")
        print("Please check your NEWS_API_KEY in .env file")

# Add this near the top of your file, after load_dotenv()
check_api_keys()

# Add these debug prints right after load_dotenv()
print("\nDebug: Checking API Keys...")
print(f"Weather API Key present: {'Yes' if os.getenv('WEATHER_API_KEY') else 'No'}")
print(f"News API Key present: {'Yes' if os.getenv('NEWS_API_KEY') else 'No'}\n")

# Download necessary NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# API Keys - These should be stored in environment variables
# Create a .env file with these keys
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')  # OpenWeatherMap API key
NEWS_API_KEY = os.getenv('NEWS_API_KEY')  # NewsAPI key

# Knowledge base for fallback responses
knowledge_base = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What's on your mind?",
        "Greetings! How may I assist you?"
    ],
    "how_are_you": [
        "I'm doing great, thank you for asking! How are you?",
        "I'm functioning well, thanks! How about you?",
        "I'm good! Always happy to chat. How are you doing?"
    ],
    "default": [
        "That's interesting. Tell me more about that.",
        "I'm not sure I fully understand. Could you tell me more?",
        "That's a good point. What else is on your mind?"
    ],
    "question": [
        "That's a good question. Let me think about that.",
        "Interesting question. I'd need to know more to give you a complete answer.",
        "I'm not entirely sure, but I'll try to help you find out."
    ]
}


# Setup function for improved language model
def setup_transformer_model():
    try:
        print("Loading GPT-2 Medium model (this may take a moment)...")
        model_name = "gpt2-medium"

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        tokenizer.pad_token = tokenizer.eos_token

        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Could not load language model ({str(e)})")
        print("Continuing with basic functionality...")
        return None, None


# Function to generate response using transformer model
def generate_transformer_response(input_text, model, tokenizer, conversation_history=None):
    try:
        # Create a prompt with the conversation history
        if conversation_history and len(conversation_history) >= 2:
            # Format a short conversation history (last 3 exchanges)
            formatted_history = []
            for i in range(min(6, len(conversation_history)), 0, -2):
                if i >= 2:
                    user_msg = conversation_history[-i]
                    bot_msg = conversation_history[-i + 1] if i > 1 else ""
                    formatted_history.append(f"Human: {user_msg}\nBot: {bot_msg}")

            prompt = "\n".join(formatted_history)
            prompt += f"\nHuman: {input_text}\nBot:"
        else:
            prompt = f"Human: {input_text}\nBot:"

        # Encode the input including padding and attention mask
        encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Generate response with appropriate parameters
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 100,  # Allow for a longer response
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.85,  # Slightly higher temperature for more creative responses
        )

        # Decode the output and extract the bot's response
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Find the last "Bot:" in the sequence and extract everything after it
        parts = full_output.split("Bot:")
        if len(parts) > 1:
            response = parts[-1].strip()

            # Remove anything after "Human:" if it exists (the model might have generated a next turn)
            if "Human:" in response:
                response = response.split("Human:")[0].strip()

            return response
        else:
            return full_output

    except Exception as e:
        print(f"Error generating response: {e}")
        return None


# Improved intent recognition function
def recognize_intent(text):
    text = text.lower()

    # Check for "how are you" variations
    how_are_you_patterns = [
        "how are you",
        "how're you",
        "how you doing",
        "how do you do",
        "how are things",
        "how's it going"
    ]
    
    if any(pattern in text for pattern in how_are_you_patterns):
        return "how_are_you"

    # Check for greetings
    elif any(word in text for word in ["hello", "hi", "hey", "greetings", "sup", "yo", "hola"]):
        return "greeting"

    # Check for weather queries
    elif any(phrase in text for phrase in
             ["weather", "rain", "sunny", "hot", "cold", "temperature", "forecast", "climate", "how's it outside"]):
        return "weather"

    # Check for news queries
    elif any(phrase in text for phrase in
             ["news", "headlines", "current events", "latest", "what's happening", "update me on"]):
        return "news"

    # Check for time/date queries
    elif any(phrase in text for phrase in ["time", "date", "day", "what day", "today is", "current time"]):
        return "time"

    # Check for joke requests
    elif any(phrase in text for phrase in ["joke", "funny", "make me laugh", "tell me something funny", "humor"]):
        return "joke"

    # Check for help request
    elif any(phrase in text for phrase in ["help", "assist", "what can you do", "features", "capabilities"]):
        return "help"

    # Check for questions
    elif "?" in text or any(word in text for word in ["who", "what", "where", "when", "why", "how"]):
        return "question"

    # Default case
    else:
        return "default"


# Function to extract locations from text (improved)
def extract_location(text):
    text_lower = text.lower()

    # List of major cities (expanded)
    cities = [
        "new york", "london", "paris", "tokyo", "los angeles", "chicago", "berlin",
        "sydney", "mumbai", "beijing", "cairo", "moscow", "rome", "toronto", "mexico city",
        "izmir", "istanbul", "ankara", "san francisco", "boston", "seattle", "miami",
        "dubai", "austin", "bangkok", "seoul", "singapore", "las vegas", "phoenix",
        "delhi", "shanghai", "hong kong", "rio de janeiro", "cape town", "madrid",
        "amsterdam", "vancouver", "montreal", "dublin", "vienna", "prague", "athens",
        "houston", "dallas", "denver", "atlanta", "washington"
    ]

    # Check for direct city mentions
    for city in cities:
        if city in text_lower:
            return city

    # Look for location patterns (improved regex)
    location_patterns = [
        r'in ([a-zA-Z\s]+)(?:$|[,\.\?])',  # "in X"
        r'for ([a-zA-Z\s]+)(?:$|[,\.\?])',  # "for X"
        r'weather (?:in|at|for) ([a-zA-Z\s]+)',  # "weather in/at/for X"
        r'weather of ([a-zA-Z\s]+)',  # "weather of X"
    ]

    for pattern in location_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            # Filter out common non-location words
            non_locations = ['today', 'tomorrow', 'yesterday', 'the morning', 'the evening', 'the afternoon', 'the day']
            for match in matches:
                if match.strip() not in non_locations:
                    return match.strip()

    return None


# Improved Weather API function with detailed error handling
def get_weather(location):
    if not WEATHER_API_KEY:
        return "⚠️ I can't check the weather right now because I don't have an API key configured. Please add your OpenWeather API key to the .env file."
    
    if len(WEATHER_API_KEY) < 20:  # Basic validation
        return "⚠️ The weather API key appears to be invalid. Please check your API key in the .env file."

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        print(f"\nDebug: Attempting to fetch weather for {location}")
        
        response = requests.get(url, timeout=10)
        print(f"Debug: API Response Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            weather_description = data['weather'][0]['description']
            temperature = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']

            weather_info = (
                f"🌡️ Current weather in {location.title()}:\n"
                f"• Condition: {weather_description.title()}\n"
                f"• Temperature: {temperature:.1f}°C (feels like {feels_like:.1f}°C)\n"
                f"• Humidity: {humidity}%"
            )
            return weather_info

        elif response.status_code == 401:
            print(f"Debug: Authentication failed. Response: {response.text}")
            return "⚠️ The weather API key is invalid or not yet activated. New API keys may take a few hours to activate. Please check your API key or try again later."
            
        elif response.status_code == 404:
            return f"Sorry, I couldn't find weather information for '{location}'. Please check the city name and try again."
            
        else:
            print(f"Debug: Unexpected status code {response.status_code}")
            print(f"Debug: Full error response: {response.text}")
            return f"Sorry, I encountered an issue (Error {response.status_code}) when trying to fetch weather data."

    except requests.exceptions.Timeout:
        return "The weather service is taking too long to respond. Please try again later."
    except requests.exceptions.ConnectionError:
        return "I couldn't connect to the weather service. Please check your internet connection."
    except Exception as e:
        print(f"Debug: Unexpected error: {str(e)}")
        return "I encountered an error trying to fetch the weather information."


# Improved News API function
def get_news(topic=None):
    if not NEWS_API_KEY:
        print("News API key not found. Please set the NEWS_API_KEY environment variable.")
        return "I'd like to share some news, but I'm not configured with a news API key. Please set the NEWS_API_KEY environment variable."

    try:
        if topic:
            url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}&pageSize=5&language=en&sortBy=publishedAt"
        else:
            url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}&pageSize=5"

        response = requests.get(url, timeout=10)  # Added timeout

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            if not articles:
                return f"I couldn't find any{'news articles on ' + topic if topic else ' top headlines'}. Try a different topic or check back later."

            # Format the response
            news_info = f"Here are some {'top headlines' if not topic else f'news articles about {topic}'}:\n\n"

            for i, article in enumerate(articles[:3], 1):  # Limit to 3 articles
                title = article.get('title', 'No title')
                source = article.get('source', {}).get('name', 'Unknown source')
                url = article.get('url', '')
                # Clean up title (remove source at the end if it's duplicated)
                if ' - ' + source in title:
                    title = title.replace(' - ' + source, '')

                news_info += f"{i}. {title} ({source})\n"

            return news_info
        elif response.status_code == 401:
            print("News API authentication failed. Check your API key.")
            return "I couldn't authenticate with the news service. Please check your API key."
        elif response.status_code == 429:
            return "I've reached the request limit for the news service. Please try again later."
        else:
            return f"Sorry, I encountered an issue (Error {response.status_code}) when trying to fetch the news."

    except requests.exceptions.Timeout:
        return "The news service is taking too long to respond. Please try again later."
    except requests.exceptions.ConnectionError:
        return "I couldn't connect to the news service. Please check your internet connection."
    except Exception as e:
        print(f"News API error: {e}")
        return "I encountered an error trying to fetch the news."


# Help function to explain capabilities
def get_help_info():
    return (
        "Here's what I can do for you:\n"
        "• Weather updates - Ask me about the weather in any city\n"
        "• News headlines - Ask for general news or news about a specific topic\n"
        "• Jokes - Ask me to tell you a joke\n"
        "• Time and date - Ask me for the current time or date\n"
        "• General conversation - Feel free to chat with me about other topics too\n\n"
        "Try asking things like:\n"
        "- \"What's the weather in Paris?\"\n"
        "- \"Tell me the latest news about technology\"\n"
        "- \"Tell me a joke\"\n"
        "- \"What time is it?\""
    )


# Improved joke API function
def get_joke():
    try:
        # Try JokeAPI first
        url = ("https://v2.jokeapi.dev/joke/Programming,Miscellaneous,Pun?blacklistFlags=nsfw,"
               "religious,political,racist,sexist,explicit")
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()

            if data['type'] == 'single':
                return data['joke']
            else:
                return f"{data['setup']}\n{data['delivery']}"

        # Fall back to backup joke API
        backup_url = "https://official-joke-api.appspot.com/random_joke"
        backup_response = requests.get(backup_url, timeout=5)

        if backup_response.status_code == 200:
            joke_data = backup_response.json()
            return f"{joke_data['setup']}\n{joke_data['punchline']}"

        # If both APIs fail, use a local joke
        return random.choice([
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why was the computer cold? It left its Windows open!",
            "What's a computer's favorite snack? Microchips!",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
            "Why don't scientists trust atoms? Because they make up everything!"
        ])

    except Exception as e:
        print(f"Joke API error: {e}")
        return random.choice([
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why was the computer cold? It left its Windows open!",
            "What's a computer's favorite snack? Microchips!",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
            "Why don't scientists trust atoms? Because they make up everything!"
        ])


# Add a new class to handle conversation context
class ConversationContext:
    def __init__(self):
        self.user_name = ""
        self.conversation_history = []
        self.sentiment_scores = []
        self.last_api_call = {}  # For rate limiting
        self.conversation_state = "greeting"
    
    def add_message(self, role, message):
        self.conversation_history.append({"role": role, "message": message, "timestamp": time.time()})
        if len(self.conversation_history) > 10:  # Keep last 10 messages
            self.conversation_history.pop(0)
    
    def get_sentiment_trend(self):
        if len(self.sentiment_scores) < 2:
            return "neutral"
        avg = sum(self.sentiment_scores[-3:]) / len(self.sentiment_scores[-3:])
        if avg > 0.2:
            return "positive"
        elif avg < -0.2:
            return "negative"
        return "neutral"

    def can_make_api_call(self, api_name, cooldown_seconds=60):
        current_time = time.time()
        last_call = self.last_api_call.get(api_name, 0)
        if current_time - last_call >= cooldown_seconds:
            self.last_api_call[api_name] = current_time
            return True
        return False

# Improve response generation with sentiment and context
def generate_response(user_input, context):
    try:
        # Analyze sentiment
        sentiment = sia.polarity_scores(user_input)
        context.sentiment_scores.append(sentiment['compound'])
        
        # Get intent
        intent = recognize_intent(user_input)
        
        # Handle based on sentiment trend and intent
        sentiment_trend = context.get_sentiment_trend()
        
        # Handle basic conversational intents first
        if intent == "how_are_you":
            response = random.choice(knowledge_base["how_are_you"])
            if sentiment_trend == "positive":
                response += " I'm glad you seem to be in a good mood!"
            return response
            
        elif intent == "greeting":
            greeting = random.choice(knowledge_base["greeting"])
            if context.conversation_history:  # Not the first interaction
                greeting = "Nice to hear from you again! " + greeting
            return greeting
            
        # Handle other intents
        elif intent == "weather":
            if not context.can_make_api_call("weather", 30):
                return "I just checked the weather recently. Please wait a moment before asking again."
                
            location = extract_location(user_input)
            if location:
                return get_weather(location)
            return "Which city would you like to know the weather for?"
            
        elif intent == "news":
            if not context.can_make_api_call("news", 60):
                return "I recently fetched news. Please wait a minute before requesting again."
                
            topic = extract_topic(user_input)
            return get_news(topic)
            
        elif intent == "help":
            return get_help_info()
            
        elif intent == "joke":
            if sentiment_trend == "negative":
                return "I notice you might be feeling down. Here's a cheerful joke to lift your spirits:\n" + get_joke()
            return get_joke()
            
        else:
            # Use sentiment to adjust response
            if sentiment_trend == "negative":
                return random.choice([
                    "I sense you might be frustrated. How can I help make things better?",
                    "Let me know if there's anything specific I can help you with.",
                    "I'm here to help. Would you like to try something different?"
                ])
            elif sentiment_trend == "positive":
                return random.choice([
                    "I'm glad we're having a good conversation!",
                    "That's great! What else would you like to know?",
                    "Wonderful! I'm here if you need anything else."
                ])
            
            return random.choice(knowledge_base["default"])
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I encountered an error. Could you please rephrase that?"

# Add function to extract topics from news queries
def extract_topic(text):
    text = text.lower()
    # Common topic indicators
    topic_indicators = ["about", "regarding", "on", "related to"]
    
    for indicator in topic_indicators:
        if indicator in text:
            parts = text.split(indicator)
            if len(parts) > 1:
                return parts[1].strip()
    
    # Remove common words to isolate topic
    common_words = ["news", "latest", "current", "tell", "me", "the", "what", "is"]
    words = text.split()
    topic_words = [w for w in words if w not in common_words]
    
    return " ".join(topic_words) if topic_words else None

# Update main chatbot function
def enhanced_doggy_bot():
    print("Starting Advanced Doggy Chatbot...")
    
    context = ConversationContext()
    
    print("\n" + "=" * 60)
    print("🐶 \033[1mAdvanced Doggy Chatbot\033[0m")
    print("=" * 60)
    
    try:
        context.user_name = input("Enter your name: ").strip() or "User"
        print(f"\n\033[1mWelcome, {context.user_name}!\033[0m")
        print("I can provide weather updates, news, jokes, and more. How can I help you today?")
        print("(Type 'exit' or 'quit' to end our conversation)\n")
        
        while True:
            try:
                user_input = input(f"\033[94m{context.user_name}:\033[0m ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("\n\033[92mBot:\033[0m Goodbye! Have a great day!")
                    break
                
                # Add message to context
                context.add_message("user", user_input)
                
                # Generate and display response
                response = generate_response(user_input, context)
                context.add_message("bot", response)
                
                print(f"\033[92mBot:\033[0m {response}\n")
                
            except KeyboardInterrupt:
                print("\n\033[92mBot:\033[0m Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"\033[91mError:\033[0m {str(e)}")
                print("Let's continue our conversation...")
                continue
                
    except Exception as e:
        print(f"\033[91mFatal Error:\033[0m {str(e)}")
        print("Bot is shutting down...")


if __name__ == "__main__":
    enhanced_doggy_bot()


