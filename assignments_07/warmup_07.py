# --- Lesson 02 ---

from dotenv import load_dotenv
from openai import OpenAI
import os

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

client = OpenAI()
print('OpenAI client created.')

# Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Converts a Celsius temperature to Fahrenheit and return it as a formatted string.',
            'parameters': {
                'type': 'number',
                'properties': {},
                'required': [],
            },
        },
    }
]

print(f"\nQ1:\n")

cf1 = celsius_to_fahrenheit(0)
cf2 = celsius_to_fahrenheit(100)
cf3 = celsius_to_fahrenheit(-40)

print(f"{cf1}\n")
print(f"{cf2}\n")
print(f"{cf3}\n")