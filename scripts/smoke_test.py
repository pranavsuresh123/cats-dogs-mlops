import sys
import time

import requests


def wait_for_health(base_url, timeout=60):
    print("Waiting for API to become healthy...")

    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(
                f"{base_url}/health",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "healthy":
                    print("Health check: PASS")
                    return True

        except requests.RequestException:
            pass

        print("API not ready yet...")

        time.sleep(3)

    return False


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python smoke_test.py <base_url>"
        )
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")

    if not wait_for_health(base_url):
        print("Health check: FAIL")
        sys.exit(1)

    print("Checking prediction endpoint...")

    try:
        with open("test_cat.jpg", "rb") as image:
            response = requests.post(
                f"{base_url}/predict",
                files={
                    "file": (
                        "test_cat.jpg",
                        image,
                        "image/jpeg"
                    )
                },
                timeout=60
            )

        if response.status_code != 200:
            print("Prediction check failed")
            print(response.text)
            sys.exit(1)

        result = response.json()

    except Exception as exc:
        print("Prediction request failed:")
        print(exc)
        sys.exit(1)

    if result.get("class") not in {"cat", "dog"}:
        print("Invalid prediction class")
        sys.exit(1)

    probability = result.get("probability")

    if not isinstance(probability, (int, float)):
        print("Invalid probability")
        sys.exit(1)

    if not 0 <= probability <= 1:
        print("Probability outside [0, 1]")
        sys.exit(1)

    print("Prediction check: PASS")
    print(result)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()