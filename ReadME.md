# Football rules and regulation QA BOT 

### About
#### The Application is used to ask query related football. The application build using RAG system which help to improve the knowledge for domain specific by avoiding fine tuning process, which help to reduce time and resources. The UI is build using Gradio for easy deployment.


## Rquirements 

1. Install ollama
2. After installing ollama, open ollama and install the llama3.2 model by entering the command:

            ollama run llama3.2

3. You can choose any model from ollama, but don't forget to change it in the app.py file where we comment to change.

4. Now create a Python environment by running the command:

            python -m venv myenv

5. Activate the environment and install the requirements by running:

            pip install -r requirements.txt

6. Now run the application by executing:

            python app.py



## Technology or library used

1. Gradio
2. Langchain
3. Ollama

## Future upgrade

1. Adding ability to add pdf by user with new knowledge rules.
2. Error handling for clean and clear.
