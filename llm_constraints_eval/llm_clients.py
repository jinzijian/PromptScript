import subprocess
import json
from openai import OpenAI
from anthropic import Anthropic

class OllamaLLM:
    def __init__(self, model, api_url='http://localhost:11434/api/generate'):
        """
        Initializes the OllamaLLM instance.

        Args:
            model (str): The name of the Ollama model to use (e.g., "llama3.2").
            api_url (str): The URL of the Ollama API endpoint.
        """
        self.model = model
        self.api_url = api_url

    def run(self, prompt):
        return self(prompt)

    def __call__(self, prompt):
        """
        Sends a prompt to the Ollama model and retrieves the response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str or None: The response from the model as a JSON string, or None if an error occurs.
        """
        # Construct the JSON payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            # "format": "json",
            "options": {
                "temperature": 0
            },
            "stream": False
        }

        # Convert the payload to a JSON string
        payload_str = json.dumps(payload)

        # Construct the curl command
        curl_command = [
            'curl',
            # '-X', 'POST',
            self.api_url,
            # '-H', 'Content-Type: application/json',
            '-d', payload_str
        ]

        try:
            # Execute the curl command
            result = subprocess.run(
                curl_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True  # This will raise CalledProcessError for non-zero exit codes
            )
            # print(result)
            # Parse the JSON response from Ollama
            response_data = json.loads(result.stdout)

            if response_data.get("done"):
                # Extract and return the 'response' field
                print(response_data.get("response"))
                return response_data.get("response", "").strip()
            else:
                print(f"LLM response not done. Response: {response_data}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"Curl command failed with error: {e.stderr}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response from Ollama: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the OpenAIClient.

        Args:
            api_key (str): Your OpenAI API key.
            model (str): The name of the OpenAI model to use (e.g., 'gpt-3.5-turbo' or 'gpt-4').
        """
        self.client = OpenAI(
            api_key=api_key
        )
        self.model = model

    def __call__(self, messages: str) -> str:
        """
        Generate a response from the OpenAI model.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated response.
        """
        response = self.client.chat.completions.create(
            model = self.model,
            messages = messages,
            temperature = 0
        )
        return response.choices[0].message.content.strip()


class AnthropicClient:
    def __init__(self, api_key: str, model: str = "claude-v1"):
        """
        Initialize the AnthropicClient.

        Args:
            api_key (str): Your Anthropic API key.
            model (str): The name of the Anthropic model to use (e.g., 'claude-v1').
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        """
        Generate a response from the Anthropic model.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated response.
        """
        # Construct the full prompt with Anthropic's specific formatting
        response = self.client.messages.create(
            model=self.model,
            messages = [
                {
                    "role": 'user',
                    "content": prompt
                }
            ],
            max_tokens=512,
            temperature=0,
        )
        return response.content[0].text.strip()