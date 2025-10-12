FROM python:3.10-slim\nWORKDIR /app\nCOPY requirements.txt ./\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . /app\nCMD ["python", "main.py"]\n
