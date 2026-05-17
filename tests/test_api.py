import subprocess
import time
import uuid
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "docker-compose-test.yml"
BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def base_url():
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "--build", "-d"],
        cwd=str(ROOT),
        check=True,
    )

    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/", timeout=3)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "logs"],
            cwd=str(ROOT),
            check=False,
        )
        raise RuntimeError("Backend did not start in time")

    yield BASE_URL

    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"],
        cwd=str(ROOT),
        check=True,
    )


@pytest.fixture(scope="session")
def auth_headers(base_url):
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    password = "testpassword"

    register_response = requests.post(
        f"{base_url}/api/auth/register",
        json={"username": username, "password": password},
    )
    assert register_response.status_code == 200

    login_response = requests.post(
        f"{base_url}/api/auth/login",
        data={"username": username, "password": password},
    )
    assert login_response.status_code == 200

    token = login_response.json().get("access_token")
    assert token
    return {"Authorization": f"Bearer {token}"}


def test_root_endpoint_returns_hello_message(base_url):
    response = requests.get(f"{base_url}/")

    assert response.status_code == 200
    assert response.json() == {"message": "hello"}


def test_submissions_config_returns_default_char_limit(base_url):
    response = requests.get(f"{base_url}/api/submissions/config")

    assert response.status_code == 200
    assert "char_limit" in response.json()


def test_auth_register_and_login_cycle(base_url):
    username = f"testuser_{uuid.uuid4().hex[:8]}"
    password = "testpassword"

    register_response = requests.post(
        f"{base_url}/api/auth/register",
        json={"username": username, "password": password},
    )
    assert register_response.status_code == 200
    register_data = register_response.json()
    assert register_data["username"] == username
    assert isinstance(register_data["id"], int)

    login_response = requests.post(
        f"{base_url}/api/auth/login",
        data={"username": username, "password": password},
    )
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert token_data["token_type"] == "bearer"
    assert isinstance(token_data["access_token"], str)
    assert token_data["access_token"]


def test_auth_login_fails_for_invalid_credentials(base_url):
    response = requests.post(
        f"{base_url}/api/auth/login",
        data={"username": "missing", "password": "wrongpass"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_submission_create_and_retrieve(base_url, auth_headers):
    raw_code = "print('hello from pytest')"

    create_response = requests.post(
        f"{base_url}/api/submissions/",
        json={"raw_code": raw_code},
        headers=auth_headers,
        timeout=30,
    )
    assert create_response.status_code == 200
    create_data = create_response.json()
    assert create_data["raw_code"] == raw_code
    assert create_data["id"] > 0
    assert "recommendation_text" in create_data

    submission_id = create_data["id"]
    get_response = requests.get(
        f"{base_url}/api/submissions/{submission_id}",
        headers=auth_headers,
    )
    assert get_response.status_code == 200
    get_data = get_response.json()
    assert get_data["raw_code"] == raw_code
    assert "recommendation_text" in get_data
