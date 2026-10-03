import json

from src.qa_engine import QAEngine


MANIFEST_PATH = "mock_data/metadata/manifest.json"


def main():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    engine = QAEngine()

    total = len(manifest["fixtures"])
    passed = 0

    print(f"Running {total} fixtures...\n")

    for fixture in manifest["fixtures"]:
        fixture_id = fixture["id"]

        reference = "mock_data/" + fixture["reference"]
        generated = "mock_data/" + fixture["generated"]

        result = engine.analyze(reference, generated)

        actual_identity = result["identity"]["primary"]["label"]
        actual_decision = result["decision"]

        expected_identity = fixture["expected_identity"]
        expected_decision = fixture["expected_decision"]

        identity_ok = actual_identity == expected_identity
        decision_ok = actual_decision == expected_decision

        fixture_passed = identity_ok and decision_ok

        if fixture_passed:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status}] {fixture_id}")
        print(f"  Expected identity: {expected_identity}")
        print(f"  Actual identity:   {actual_identity}")
        print(f"  Expected decision: {expected_decision}")
        print(f"  Actual decision:   {actual_decision}")
        print()

    print("=" * 50)
    print(f"Fixture tests passed: {passed}/{total}")

    if passed == total:
        print("ALL FIXTURE TESTS PASSED")
    else:
        print("SOME FIXTURE TESTS FAILED")


if __name__ == "__main__":
    main()