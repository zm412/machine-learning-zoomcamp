import requests

url = "http://localhost:9696/predict"

customer = {
    "job_title": "Web Developer",
    "experience_years": 3,
    "education": "Bachelor",
    "skills": "html, javascript, python, css"
}

response = requests.post(url, json=customer).json()
print(response)

