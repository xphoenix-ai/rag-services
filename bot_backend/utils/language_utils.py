import os
import requests


def get_languages():
    """Fetch list of supported languages from the language service.
    
    Retrieves the supported languages for all services (translator, STT, TTS, etc.)
    from the languages service endpoint.
    
    Returns:
        Tuple[bool, dict]: A tuple containing:
            - success (bool): True if languages were successfully fetched, False otherwise
            - response (dict): Dictionary with language information for each service,
                or empty dict if the request failed
    """
    url = os.getenv("LANGUAGES_URL")

    try:
        response = requests.get(url)
        response = response.json()
        success = True
    except:
        response = {}
        success = False

    return success, response
