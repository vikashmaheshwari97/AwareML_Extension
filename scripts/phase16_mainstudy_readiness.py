#!/usr/bin/env python3
from awareml.studies.trust import load_phase16_protocol, validate_protocol_for_design_freeze


def main():
    protocol = load_phase16_protocol()
    errors = validate_protocol_for_design_freeze(protocol)
    print("=" * 72)
    print("Phase 16 Main Study readiness")
    print("=" * 72)
    print("Protocol status:", protocol.get("status"))
    print("Trust measure:", (protocol.get("trust_measure") or {}).get("status"))
    print("Power calculation:", (protocol.get("power_calculation") or {}).get("status"))
    print("Ethics:", (protocol.get("ethics") or {}).get("status"))
    print("Participant materials:", (protocol.get("participant_materials") or {}).get("status"))
    print("Secondary post-study:", (protocol.get("secondary_poststudy") or {}).get("status"))
    print("")
    if errors:
        print("MAIN STUDY: NOT READY")
        for i, error in enumerate(errors, 1):
            print("  {}. {}".format(i, error))
    else:
        print("Protocol fields are ready for design freeze.")
        print("Next: python -m scripts.freeze_phase16_design")
    print("=" * 72)


if __name__ == "__main__":
    main()
