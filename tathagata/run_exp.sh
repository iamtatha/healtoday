python3 -W ignore ollama_agent.py llama3.2 llama3.2 > /dev/null 2>&1
echo "Done llama3.2"
python3 -W ignore ollama_agent.py llama3.1:8b llama3.1:8b > /dev/null 2>&1
echo "Done llama3.1:80b"
python3 -W ignore ollama_agent.py mistral:7b mistral:7b > /dev/null 2>&1
echo "Done mistral:7b"
python3 -W ignore gpt_agent.py gpt-3.5-turbo gpt-3.5-turbo > /dev/null 2>&1
echo "Done gpt-3.5-turbo"
python3 -W ignore gpt_agent.py gpt-4o-mini gpt-4o-mini > /dev/null 2>&1
echo ""Done gpt-4o-mini
python3 -W ignore gpt_agent.py gpt-4o gpt-4o > /dev/null 2>&1
echo "Done gpt-4o"