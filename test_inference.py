
import os
import requests
import getpass

API_URL = "https://crop-disease-inference.onrender.com/predict"

# Enter your API key privately when prompted
api_key = getpass.getpass("Enter your inference API key: ")

# Enter the path of a leaf image
image_path = input("Enter leaf image path: ").strip().strip('"')

if not os.path.isfile(image_path):
    print("Image file not found!")
    raise SystemExit(1)

headers = {
    "X-API-Key": api_key
}

try:
    with open(image_path, "rb") as image:
        files = {
            "image": (os.path.basename(image_path), image)
        }

        response = requests.post(
            API_URL,
            headers=headers,
            files=files,
            timeout=300
        )

    print("\nHTTP Status:", response.status_code)
    print("Response:", response.text)

except requests.RequestException as error:
    print("Request failed:", error)
