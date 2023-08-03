import argparse
import configparser
import logging
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    filename="app.log", filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

import openai

class PythonToUMLConverter:
    def __init__(
        self, project_dir: Path, uml_dir: Path,
        token_limit: int = 3500, max_attempts: int = 3
    ):
        self.project_dir = project_dir
        self.uml_dir = uml_dir
        self.token_limit = token_limit
        self.max_attempts = max_attempts
        self._validate_api_key()

    def _validate_api_key(self):
        try:
            # A simple test request to the OpenAI API
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "test"},
                ],
                max_tokens=1
            )
        except openai.error.AuthenticationError:
            logging.error("AuthenticationError: The OpenAI API key is not valid.")
            raise ValueError("AuthenticationError: The OpenAI API key is not valid.")
        except openai.error.APIError as e:
            if "Forbidden" in str(e):
                logging.error("APIError: The OpenAI API key does not have access to the GPT-3.5-turbo engine.")
                raise ValueError("APIError: The OpenAI API key does not have access to the GPT-3.5-turbo engine.")
            else:
                logging.error("APIError: Failed to make a request to the OpenAI API: %s", e)
                raise e

    def _scan_directory(self) -> str:
        python_code = ""
        for file in self.project_dir.rglob('*.py'):
            with open(file, 'r') as f:
                python_code += f.read() + '\n\n'
        if not python_code:
            logging.error('No Python files found in directory.')
        return python_code

    def _convert_code_to_uml(self, python_code: str) -> Optional[str]:
        chunk_size = self.token_limit - 200  # Save some tokens for the prompt and API overhead
        chunks = [python_code[i:i+chunk_size] for i in range(0, len(python_code), chunk_size)]
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant and brilliant programmer. You are shown a programming project in sections"}
        ]
        
        # Add each chunk to the conversation
        for chunk in chunks:
            conversation.append({"role": "user", "content": self._generate_prompt(chunk)})

        # Add the instruction to generate a UML diagram
        conversation.append({"role": "user", "content": "given all of that python code, generate a plantUML code for a UML class diagram for the whole coding project"})
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=conversation,
                temperature=0.5,
                max_tokens=2000
            )
            # Log the entire response
            logging.info(f'OpenAI API response: {response}')
            # Extract UML code from the response
            uml_code = self._extract_uml_code(response.choices[0].message["content"])
            return uml_code if uml_code else None
        except Exception as e:
            logging.error('OpenAI API call failed with exception: %s', e)
            return None
        except openai.error.APIError as e:
            logging.error('OpenAI APIError occurred: %s', str(e))
            return None
        except openai.error.AuthenticationError as e:
            logging.error('OpenAI AuthenticationError occurred: %s', str(e))
            return None
        except openai.error.RateLimitError as e:
            logging.error('OpenAI RateLimitError occurred: %s', str(e))
            return None
        except Exception as e:
            logging.error('An unexpected error occurred when making the OpenAI API call: %s', str(e))
            return None

    def _extract_uml_code(self, message: str) -> Optional[str]:
        logging.info(f'Extracting UML code from: {message}')
        try:
            start = message.index('@startuml')
            end = message.index('@enduml') + len('@enduml')
            return message[start:end]
        except ValueError:
            return None


    def _generate_prompt(self, chunk: str) -> str:
        prompt = (chunk)
        return prompt

    def _save_to_file(self, uml_code: str) -> Path:
        timestamp = str(int(time.time()))
        folder_name = 'uml_code_' + timestamp
        print(folder_name)
        self.uml_dir.joinpath(folder_name).mkdir(parents=True, exist_ok=True) 

        filename = self.uml_dir.joinpath(folder_name, 'uml_code.txt')
        try:
            with open(filename, 'w') as f:
                f.write(uml_code)
            logging.info(f'UML code saved to {filename}')
            return filename
        except Exception as e:
            logging.error('Failed to write UML code to file with exception: %s', e)
            return None

    def convert(self):
        python_code = self._scan_directory()
        if python_code is None:
            return

        for attempt in range(self.max_attempts):
            try:
                uml_code = self._convert_code_to_uml(python_code)
                if uml_code is None:
                    raise Exception("Failed to convert code to UML")

                uml_file = self._save_to_file(uml_code)
                if uml_file is None:
                    raise Exception("Failed to save UML code to file")

                break  # If everything went fine, break the loop
            except Exception as e:
                logging.error('Attempt %d: An error occurred during conversion: %s', attempt + 1, e)
                if attempt == self.max_attempts - 1:  # If this was the last attempt
                    logging.error('Failed to create UML diagram after %d attempts', self.max_attempts)


def load_config(config_path: Path) -> dict:
    config = configparser.ConfigParser()
    config.read(config_path)

    config_dict = {
        "project_dir": Path(config["DEFAULT"]["ProjectDirectory"]),
        "uml_dir": Path(config["DEFAULT"]["UMLDirectory"]),
        "openai_key": config["DEFAULT"]["OpenAIKey"]
    }
    openai.api_key = config_dict["openai_key"]

    return config_dict

def main():
    parser = argparse.ArgumentParser(description="Convert Python code to UML diagrams.")
    parser.add_argument(
        "--config", "-c", type=Path, default="config.ini",
        help="Path to the configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    converter = PythonToUMLConverter(
        config["project_dir"], config["uml_dir"]
    )
    converter.convert()

if __name__ == "__main__":
    main()
