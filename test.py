from ollama import ChatResponse, chat


def add_two_numbers(a: int, b: int) -> int:
  return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
  return int(a) - int(b)

def query(message):
  messages = [{'role': 'user', 'content': message}]
  print('Prompt:', messages[0]['content'])

  available_functions = {
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
  }

  response: ChatResponse = chat(
    'qwen3:0.6b',
    messages=messages,
    tools=[add_two_numbers, subtract_two_numbers],
  )

  if response.message.tool_calls:
    for tool in response.message.tool_calls:
      if function_to_call := available_functions.get(tool.function.name):
        output = function_to_call(**tool.function.arguments)
      else:
        print('Function', tool.function.name, 'not found')
        return f"Error (missing tool: {tool.function_name})"

  if response.message.tool_calls:
    messages.append(response.message)
    messages.append({'role': 'tool', 'content': str(output), 'tool_name': tool.function.name})

    final_response = chat('qwen3:0.6b', messages=messages)
    return final_response.message.content
  else:
    return "No answer"

print(query("What's 3 + 17?"))
print(query("What's 3 - 17?"))