import sys
import requests


def run_tests(base_url: str):
    base_url = base_url.rstrip("/")
    all_passed = True

    # Test 1: Health
    try:
        r = requests.get(f"{base_url}/health")
        if r.status_code == 200 and r.json().get("status") == "ok":
            print("PASS /health")
        else:
            print("FAIL /health")
            all_passed = False
    except Exception as e:
        print(f"FAIL /health — {e}")
        all_passed = False

    # Test 2: Predict
    try:
        payload = {
            "age": 30,
            "scholarship": 0,
            "hypertension": 0,
            "diabetes": 0,
            "alcoholism": 0,
            "handicap": 0,
            "sms_received": 1,
            "days_in_advance": 5,
            "appt_day_of_week": 2,
        }
        r = requests.post(f"{base_url}/predict", json=payload)
        if r.status_code == 200 and "risk_level" in r.json():
            print("PASS /predict")
        else:
            print("FAIL /predict")
            all_passed = False
    except Exception as e:
        print(f"FAIL /predict — {e}")
        all_passed = False

    # Test 3: Stats
    try:
        r = requests.get(f"{base_url}/stats")
        if r.status_code == 200 and "total_predictions" in r.json():
            print("PASS /stats")
        else:
            print("FAIL /stats")
            all_passed = False
    except Exception as e:
        print(f"FAIL /stats — {e}")
        all_passed = False

    if all_passed:
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <base_url>")
        sys.exit(1)
    run_tests(sys.argv[1])