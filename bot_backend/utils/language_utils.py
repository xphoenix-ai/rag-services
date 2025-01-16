import os
import requests


def get_languages():
    url = os.getenv("LANGUAGES_URL")

    try:
        response = requests.get(url)
        response = response.json()
        success = True
    except:
        response = {}
        success = False

    return success, response
