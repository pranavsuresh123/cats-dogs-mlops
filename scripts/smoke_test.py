import sys
import requests


def main():
    if len(sys.argv) != 2:
        print("Usage: python smoke_test.py <base_url>")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")

    print("Checking health endpoint...")

    response = requests.get(
        f"{base_url}/health",
        timeout=10
    )

    if response.status_code != 200:
        print("Health check failed:", response.status_code)
        sys.exit(1)

    if response.json().get("status") != "healthy":
        print("Invalid health response")
        sys.exit(1)

    print("Health check: PASS")

    print("Checking prediction endpoint...")

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
            timeout=30
        )

    if response.status_code != 200:
        print("Prediction check failed")
        print(response.text)
        sys.exit(1)

    result = response.json()

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