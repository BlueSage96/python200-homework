# --- Lesson 02 ---
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

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
                'type': 'object',
                'properties': {'celsius' : {
                    'type':'number'
                    } },
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

# Q2
# Yes because the model needs to call the tool that uses celsius_to_fahrenheit()
# Two API calls will be made to answer the query: one where the model asks for a 
# tool, andone where the model uses the tool result to answer.

print(f"\nQ2:\n")

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

convert = run_agent("Covert 100 degrees Celsius to Fahrenheit")
print("Conversion",convert)

# There were two API calls, but the second one had an error as it didn't recognize 
# 'celsius_to_fahrenheit'. As a result, the LLM told the user how to do the conversion 
# instead of outright doing the calculation itself.

print(f"\nQ3:\n")

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
        'type': 'function',
        'function' : {
            'name': 'celsius_to_fahrenheit',
            'description': 'Converts a Celsius temperature to Fahrenheit and return it as a formatted string.',
            'parameters': {
                'type': 'object',
                'properties': {'celsius': {
                        'type': 'number'
                    }},
                'required': []
            }
        }
    }
]
print('Tools list defined with one tool: get_current_time')

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("\nFirst response received from model...\n")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("\nAgentic mode engaged...\n")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: celsius_to_fahrenheit
            if function_name == 'celsius_to_fahrenheit':
                tool_result =  celsius_to_fahrenheit(100)
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("\nSecond response received from model...\n")
        print(f"\n{second_response}]\n")

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("\nNo tools needed....\n")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print(f"\nResponse A: {response_a}\n")
# The celsius_to_fahrenheit tool was called because it was doing a celsius to fahrenheit conversion.

response_b = run_agent("What is the boiling point of water in plain English?")
print(f"\nResponse B: {response_b}\n")
# No tools were needed to respond. There was no calculations needed.