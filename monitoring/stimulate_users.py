import requests
import pandas as pd
import random
import string
from tqdm import tqdm

def generate_random_password(length=8):
    """Generate a random password."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def submit_code_as_user(user_name, code, credentials_df, api_base_url="http://localhost:8000"):
    """
    Connect to the Python Editor app as a user, register/login, and submit code.

    Args:
        user_name (str): The username to use.
        credentials_df (pd.DataFrame): DataFrame with 'user_name' and 'password' columns.
        code (str): The Python code to submit.
        api_base_url (str): Base URL of the API.

    Returns:
        dict: Response from the submission API.
    """
    # Check if user exists in credentials_df
    user_row = credentials_df[credentials_df['user_name'] == user_name]
    if not user_row.empty:
        password = user_row['password'].iloc[0]
    else:
        # Register new user
        password = generate_random_password()
        register_url = f"{api_base_url}/api/auth/register"
        register_data = {"username": user_name, "password": password}
        response = requests.post(register_url, json=register_data)
        if response.status_code != 200:
            raise Exception(f"Registration failed: {response.text}")
        # Add to credentials_df
        new_row = pd.DataFrame({"user_name": [user_name], "password": [password]})
        credentials_df = pd.concat([credentials_df, new_row], ignore_index=True)

    # Login
    login_url = f"{api_base_url}/api/auth/login"
    login_data = {"username": user_name, "password": password}
    response = requests.post(login_url, data=login_data)
    if response.status_code != 200:
        raise Exception(f"Login failed: {response.text}")
    token = response.json()["access_token"]

    # Submit code
    submit_url = f"{api_base_url}/api/submissions/"
    headers = {"Authorization": f"Bearer {token}"}
    submit_data = {"raw_code": code}
    response = requests.post(submit_url, json=submit_data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Submission failed: {response.text}")
    return credentials_df, response.json()


def stimulate_users(users_df, api_base_url="http://localhost:8000"):
    if requests.get(api_base_url).status_code != 200:
        raise Exception(f"Failed to connect to API at {api_base_url}")
    
    credentials_df = pd.DataFrame({"user_name": [], "password": []})

    for row in tqdm(users_df.itertuples(), total=len(users_df)):
        user_name = row.repo_name.split("/")[0]
        code = row.text
        credentials_df, response = submit_code_as_user(user_name, code, credentials_df, api_base_url)
    return credentials_df