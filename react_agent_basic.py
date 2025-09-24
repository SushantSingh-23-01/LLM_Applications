from typing import Any, List
import re
import json
import ollama
from ddgs import DDGS
import colorama
from colorama import Style, Fore
colorama.init(autoreset=True)

class Config:
    def __init__(self) -> None:
        self.model_name = 'qwen3:1.7b'
        
        self.temperature = 0.1

class DateTimeTool:
    def __init__(self):
        self.parameters = [
            {
                'name': 'format_string',
                'type': 'string',
                'description': 'Datetime format. E.g. %Y-%m-%d or %Y-%m-%d-%H-%M-%S'
            }
        ]
    
    def __call__(self, format_string: str) -> str:
        import datetime
        return datetime.datetime.now().strftime(format_string)

class WebSearchTool:
    def __init__(self):
        self.parameters = [
            {
                'name': 'query',
                'type': 'string',
                'description': (
                    'Query to be used in a web search engine. '
                    'Web searches might contain old and new data. ' 
                    'Example: Who is current president of U.S.A.?'
                    )
            }
        ]
    
    def __call__(self, query:str) -> Any:
        results = ''
        web_search_data = DDGS().text(query, max_results=3)
        for search in web_search_data:
            results += f'\n- *({search.get('href', 'N.A.')})*: {search.get('body', 'No text was available.')}'
        return results
        
class CalculatorTool:
    def __init__(self) -> None:
        self.parameters = [
            {
                'name': 'var1',
                'type': 'float',
                'description': 'Variable 1.'
            },
            {
                'name': 'var2',
                'type': 'float',
                'description': 'Variable 2.'
            },
            {
                'name': 'operation',
                'type': 'str',
                'description': 'Choose one of the following operations: addition, multiplication, division or exponent'
            },
        ]
    
    def __call__(self, var1:float, var2:float, operation:str) -> Any:
        if operation == 'addition':
            return str(round(var1 + var2, 4))
        elif operation == 'multiplication':
            return str(round(var1 * var2, 2))
        elif operation == 'division':
            if var2 == 0:
                return 'Variable 2 cannot be 0.'
            return str(round(var1 / var2, 2))
        elif operation == 'exponent':
            return str(round(var1 ** var2, 2))
        
class SimpleReactAgent:
    def _get_tools_str(self, tools: List[Any])->str:
        tool_def = ''
        for tool in tools:
            tool_def += f'- {tool.__class__.__name__} | '
            for param in tool.parameters:
                tool_def += f'({param['name']}: {param['type']}) -> {param['description']} | '
            tool_def += '\n'
        return tool_def

    def _parse_json(self, response):
        json_str = None
        try: 
            match = re.search(r'(\{.*\})', response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response.strip()
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode JSON. Error: {e}. Raw response: {response}")
        except (KeyError, IndexError, TypeError) as e:
            print(f"ERROR: Unexpected JSON structure or key not found. Error: {e}. Raw response: {response}")
        return json_str
    
    def _ollama_response(self, system_prompt:str, user_msg:str, history:List, config:Any):
        ollama_history = (
            [{'role':'system', 'content':system_prompt}] +
            history +
            [{'role':'user', 'content':user_msg}]
        )
        response = ollama.chat(
            model=config.model_name, 
            messages=ollama_history,
            options={
                'temperature': config.temperature
            }
            )['message']['content']
        return response
    
    def run(self, user_msg:str, tools:List, config: Any, num_steps:int = 5)->str:
        tools_info = self._get_tools_str(tools)
        
        system_prompt = (
            'You are a smart agent who answers user query using the tools provided below. Tools:'
            f'\n{tools_info}'
            '- AnswerTool | (final_answer : string) -> If you have an answer and require no further tools.\n\n'
            'Provide output in JSON format **only**:'
            '```json\n{"tool": "Name of the tool", "args": {"args1": "value1", "args2": "value2"}}\n```'
        )
        print(system_prompt)
        history = []
        for i in range(num_steps):
            print('='*50, f'\n{'':<22}Step-{i}\n', '='*50)
            response = self._ollama_response(system_prompt, user_msg, history, config)
            json_str = self._parse_json(response)
            print(f'{Fore.YELLOW}Raw output: {json_str}{Style.RESET_ALL}')
            if json_str:
                data = json.loads(json_str)
                if data is None:
                    return "Error: Counld not parse agent's response"
                
                tool_name = data.get('tool')
                args = data.get('args', {})
                
                if tool_name == 'AnswerTool':
                    print(f'{Fore.GREEN}Final Answer: {args.get('final_answer', '')}{Style.RESET_ALL}')
                    return args.get('final_answer', '')
                
                if tool_name:
                    tool_to_use = next((t for t in tools if t.__class__.__name__ == tool_name), None)
                    if tool_to_use:
                        print(f'{Fore.BLUE}Executing Tool: {tool_name} with args: {args}{Style.RESET_ALL}') 
                        
                        try:
                            tool_result = tool_to_use(**args)
                            print(f'{Fore.MAGENTA}Tool result: {tool_result}{Style.RESET_ALL}')
                            
                            # Update history with the action and observation for the next step
                            # Note: In agents the "user" role is a placeholder for external input whether it's the initial query or the result of a tool execution.
                            history.append({'role': 'user', 'content': json.dumps({'tool_call': {'tool': tool_name, 'args': args}})})
                            history.append({'role': 'assistant', 'content': json.dumps({'tool_output': tool_result})})
                            
                            # The user message for the next step becomes the tool's output
                            user_msg = f'Observtion: {tool_result}'
                        except Exception as e:
                            print(f'Error: Tool execution failed: {e}')
                else:
                    print(f"{Fore.RED}ERROR: Tool '{tool_name}' not found.{Style.RESET_ALL}")
                    return f"Error: Tool '{tool_name}' not found."
            else:
                print("Agent did not specify a tool or a final answer.")
                return "Error: Agent could not decide on a tool or final answer."
        
        return "Maximum number of steps reached without a final answer."
