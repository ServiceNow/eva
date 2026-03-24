"""Test that all airline scenario database JSONs conform to the schema documented in docs/airline_database_tool_schema.md.

Run with:
    pytest tests/unit/test_scenario_db_schema.py -v
"""

import json
import re
from pathlib import Path

import pytest

SCENARIO_DB_DIR = Path(__file__).resolve().parents[2] / "data" / "airline_scenarios"

FARE_CLASSES = {"basic_economy", "main_cabin", "premium_economy", "business", "first"}
RESERVATION_STATUSES = {"confirmed", "changed", "cancelled"}
FARE_TYPES = {"refundable", "non_refundable"}
BOOKING_STATUSES = {"confirmed", "cancelled"}
JOURNEY_STATUSES = {"scheduled", "on_time", "delayed", "cancelled", "departed"}
DISRUPTION_TYPES = {"cancellation", "delay", "schedule_change"}
CAUSE_CATEGORIES = {"weather", "airline_fault", "mechanical"}
SEAT_TYPES = {"window", "aisle", "middle"}
MEAL_TYPES = {
    "vegetarian",
    "vegan",
    "kosher",
    "halal",
    "gluten_free",
    "diabetic",
    "low_sodium",
    "child",
    "hindu",
    "none",
    "standard",
}
SEAT_PREFERENCES = {"window", "aisle", "middle", "no_preference"}

# Regex patterns
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

SCENARIO_FILES = sorted(SCENARIO_DB_DIR.glob("*.json"))


def _id(path: Path) -> str:
    return path.stem


@pytest.fixture(params=SCENARIO_FILES, ids=[_id(f) for f in SCENARIO_FILES])
def scenario_db(request) -> tuple[str, dict]:
    """Load a scenario database JSON and return (filename, data)."""
    path: Path = request.param
    with open(path) as f:
        data = json.load(f)
    return path.name, data


REQUIRED_TOP_KEYS = {"_current_date", "reservations", "journeys", "disruptions"}
OPTIONAL_TOP_KEYS = {"travel_credits", "meal_vouchers", "hotel_vouchers", "refunds"}
ALL_TOP_KEYS = REQUIRED_TOP_KEYS | OPTIONAL_TOP_KEYS


class TestTopLevelStructure:
    def test_required_keys_present(self, scenario_db):
        name, data = scenario_db
        missing = REQUIRED_TOP_KEYS - set(data.keys())
        assert not missing, f"{name}: missing required top-level keys: {missing}"

    def test_no_unexpected_keys(self, scenario_db):
        name, data = scenario_db
        unexpected = set(data.keys()) - ALL_TOP_KEYS
        assert not unexpected, f"{name}: unexpected top-level keys: {unexpected}"

    def test_current_date_format(self, scenario_db):
        name, data = scenario_db
        assert isinstance(data["_current_date"], str), f"{name}: _current_date must be str"
        assert DATE_RE.match(data["_current_date"]), (
            f"{name}: _current_date '{data['_current_date']}' doesn't match YYYY-MM-DD"
        )

    def test_tables_are_dicts(self, scenario_db):
        name, data = scenario_db
        for key in ["reservations", "journeys", "disruptions"]:
            assert isinstance(data[key], dict), f"{name}: '{key}' must be a dict"
        for key in OPTIONAL_TOP_KEYS:
            if key in data:
                assert isinstance(data[key], dict), f"{name}: '{key}' must be a dict"


RESERVATION_FIELDS = {
    "confirmation_number": str,
    "status": str,
    "booking_date": str,
    "fare_type": str,
    "passengers": list,
    "bookings": list,
    "ancillaries": dict,
}

PASSENGER_FIELDS = {
    "passenger_id": str,
    "first_name": str,
    "last_name": str,
    "ticket_number": str,
    "email": str,
    "phone": str,
    "elite_status": (str, type(None)),
    "meal_preference": (str, type(None)),
    "seat_preference": (str, type(None)),
}

BOOKING_JOURNEY_FIELDS = {
    "journey_id": str,
    "fare_class": str,
    "fare_paid": (int, float),
    "status": str,
    "segments": list,
}

BOOKING_SEGMENT_FIELDS = {
    "flight_number": str,
    "date": str,
    "fare_paid": (int, float),
    "seat": (str, type(None)),
    "bags_checked": int,
    "meal_request": (str, type(None)),
}

ANCILLARY_FIELDS = {
    "seat_selection_fee": (int, float),
    "bags_fee": (int, float),
}


class TestReservations:
    def test_key_matches_confirmation_number(self, scenario_db):
        name, data = scenario_db
        for key, res in data["reservations"].items():
            assert key == res.get("confirmation_number"), (
                f"{name}: reservation key '{key}' != confirmation_number '{res.get('confirmation_number')}'"
            )

    def test_reservation_fields(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for field, expected_type in RESERVATION_FIELDS.items():
                assert field in res, f"{name}/{conf}: missing field '{field}'"
                assert isinstance(res[field], expected_type), (
                    f"{name}/{conf}: '{field}' should be {expected_type}, got {type(res[field])}"
                )

    def test_reservation_status_enum(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            assert res["status"] in RESERVATION_STATUSES, f"{name}/{conf}: invalid status '{res['status']}'"

    def test_reservation_fare_type_enum(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            assert res["fare_type"] in FARE_TYPES, f"{name}/{conf}: invalid fare_type '{res['fare_type']}'"

    def test_booking_date_format(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            assert ISO_DATETIME_RE.match(res["booking_date"]), (
                f"{name}/{conf}: booking_date '{res['booking_date']}' doesn't match ISO datetime"
            )

    def test_ancillaries_fields(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            anc = res["ancillaries"]
            for field, expected_type in ANCILLARY_FIELDS.items():
                assert field in anc, f"{name}/{conf}: ancillaries missing '{field}'"
                assert isinstance(anc[field], expected_type), (
                    f"{name}/{conf}: ancillaries.{field} should be {expected_type}, got {type(anc[field])}"
                )

    def test_passengers_not_empty(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            assert len(res["passengers"]) > 0, f"{name}/{conf}: passengers list is empty"

    def test_passenger_fields(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, pax in enumerate(res["passengers"]):
                for field, expected_type in PASSENGER_FIELDS.items():
                    assert field in pax, f"{name}/{conf}/passenger[{i}]: missing '{field}'"
                    assert isinstance(pax[field], expected_type), (
                        f"{name}/{conf}/passenger[{i}]: '{field}' should be {expected_type}, got {type(pax[field])}"
                    )

    def test_passenger_ids_unique_across_reservations(self, scenario_db):
        """All passenger_ids must be globally unique within a scenario, even across different reservations."""
        name, data = scenario_db
        seen: dict[str, str] = {}  # passenger_id -> confirmation_number
        for conf, res in data["reservations"].items():
            for pax in res["passengers"]:
                pid = pax["passenger_id"]
                assert pid not in seen, (
                    f"{name}: passenger_id '{pid}' appears in both reservation '{seen[pid]}' and '{conf}'"
                )
                seen[pid] = conf

    def test_passenger_id_format(self, scenario_db):
        """All passenger_ids must match the format PAX followed by exactly 3 digits (e.g. PAX001)."""
        name, data = scenario_db
        pax_re = re.compile(r"^PAX\d{3}$")
        for conf, res in data["reservations"].items():
            for pax in res["passengers"]:
                pid = pax["passenger_id"]
                assert pax_re.match(pid), f"{name}/{conf}: passenger_id '{pid}' doesn't match expected format PAXnnn"

    def test_bookings_not_empty(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            assert len(res["bookings"]) > 0, f"{name}/{conf}: bookings list is empty"

    def test_booking_journey_fields(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                for field, expected_type in BOOKING_JOURNEY_FIELDS.items():
                    assert field in bk, f"{name}/{conf}/booking[{i}]: missing '{field}'"
                    assert isinstance(bk[field], expected_type), (
                        f"{name}/{conf}/booking[{i}]: '{field}' should be {expected_type}, got {type(bk[field])}"
                    )

    def test_booking_fare_class_enum(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                assert bk["fare_class"] in FARE_CLASSES, (
                    f"{name}/{conf}/booking[{i}]: invalid fare_class '{bk['fare_class']}'"
                )

    def test_booking_status_enum(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                assert bk["status"] in BOOKING_STATUSES, f"{name}/{conf}/booking[{i}]: invalid status '{bk['status']}'"

    def test_booking_segment_fields(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                for j, seg in enumerate(bk["segments"]):
                    for field, expected_type in BOOKING_SEGMENT_FIELDS.items():
                        assert field in seg, f"{name}/{conf}/booking[{i}]/seg[{j}]: missing '{field}'"
                        assert isinstance(seg[field], expected_type), (
                            f"{name}/{conf}/booking[{i}]/seg[{j}]: "
                            f"'{field}' should be {expected_type}, "
                            f"got {type(seg[field])}"
                        )

    def test_booking_segment_date_format(self, scenario_db):
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                for j, seg in enumerate(bk["segments"]):
                    assert DATE_RE.match(seg["date"]), (
                        f"{name}/{conf}/booking[{i}]/seg[{j}]: date '{seg['date']}' doesn't match YYYY-MM-DD"
                    )

    def test_booking_journey_references_valid_journey(self, scenario_db):
        name, data = scenario_db
        journey_ids = set(data["journeys"].keys())
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                assert bk["journey_id"] in journey_ids, (
                    f"{name}/{conf}/booking[{i}]: journey_id '{bk['journey_id']}' not found in journeys table"
                )

    def test_confirmation_numbers_are_uppercase(self, scenario_db):
        """Confirmation numbers (dict keys) must be uppercase since the tools code does .upper() before lookup."""
        name, data = scenario_db
        for key in data["reservations"]:
            assert key == key.upper(), f"{name}: reservation key '{key}' is not uppercase"


JOURNEY_FIELDS = {
    "journey_id": str,
    "date": str,
    "origin": str,
    "destination": str,
    "num_stops": int,
    "total_duration_minutes": int,
    "status": str,
    "bookable": bool,
    "fares": dict,
    "segments": list,
}

JOURNEY_SEGMENT_FIELDS = {
    "segment_number": int,
    "flight_number": str,
    "origin": str,
    "destination": str,
    "scheduled_departure": str,
    "scheduled_arrival": str,
    "origin_utc_offset": int,
    "destination_utc_offset": int,
    "duration_minutes": int,
    "aircraft_type": str,
    "status": str,
    "delay_minutes": (int, type(None)),
    "delay_reason": (str, type(None)),
    "cancellation_reason": (str, type(None)),
    "gate": (str, type(None)),
    "available_seats": dict,
    "available_seat_types": dict,
    "fares": dict,
}


class TestJourneys:
    def test_key_matches_journey_id(self, scenario_db):
        name, data = scenario_db
        for key, journey in data["journeys"].items():
            assert key == journey.get("journey_id"), (
                f"{name}: journey key '{key}' != journey_id '{journey.get('journey_id')}'"
            )

    def test_journey_fields(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for field, expected_type in JOURNEY_FIELDS.items():
                assert field in journey, f"{name}/{jid}: missing field '{field}'"
                assert isinstance(journey[field], expected_type), (
                    f"{name}/{jid}: '{field}' should be {expected_type}, got {type(journey[field])}"
                )

    def test_journey_date_format(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            assert DATE_RE.match(journey["date"]), f"{name}/{jid}: date '{journey['date']}' doesn't match YYYY-MM-DD"

    def test_journey_status_enum(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            assert journey["status"] in JOURNEY_STATUSES, f"{name}/{jid}: invalid status '{journey['status']}'"

    def test_journey_fares_keys(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            fares = journey["fares"]
            assert set(fares.keys()) == FARE_CLASSES, (
                f"{name}/{jid}: fares keys {set(fares.keys())} != expected {FARE_CLASSES}"
            )

    def test_journey_fares_values(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for fare_class, price in journey["fares"].items():
                assert price is None or isinstance(price, (int, float)), (
                    f"{name}/{jid}: fares['{fare_class}'] should be float or None, got {type(price)}"
                )

    def test_journey_segments_not_empty(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            assert len(journey["segments"]) > 0, f"{name}/{jid}: segments list is empty"

    def test_journey_num_stops_consistent(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            expected_stops = len(journey["segments"]) - 1
            assert journey["num_stops"] == expected_stops, (
                f"{name}/{jid}: num_stops={journey['num_stops']} but "
                f"has {len(journey['segments'])} segments "
                f"(expected {expected_stops} stops)"
            )

    def test_journey_segment_fields(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                for field, expected_type in JOURNEY_SEGMENT_FIELDS.items():
                    assert field in seg, f"{name}/{jid}/seg[{i}]: missing '{field}'"
                    assert isinstance(seg[field], expected_type), (
                        f"{name}/{jid}/seg[{i}]: '{field}' should be {expected_type}, got {type(seg[field])}"
                    )

    def test_journey_segment_numbers_sequential(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            seg_nums = [s["segment_number"] for s in journey["segments"]]
            expected = list(range(1, len(journey["segments"]) + 1))
            assert seg_nums == expected, f"{name}/{jid}: segment_numbers {seg_nums} != expected {expected}"

    def test_journey_segment_status_enum(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                assert seg["status"] in JOURNEY_STATUSES, f"{name}/{jid}/seg[{i}]: invalid status '{seg['status']}'"

    def test_journey_segment_time_formats(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                assert TIME_RE.match(seg["scheduled_departure"]), (
                    f"{name}/{jid}/seg[{i}]: scheduled_departure '{seg['scheduled_departure']}' doesn't match HH:MM"
                )
                assert TIME_RE.match(seg["scheduled_arrival"]), (
                    f"{name}/{jid}/seg[{i}]: scheduled_arrival '{seg['scheduled_arrival']}' doesn't match HH:MM"
                )

    def test_journey_segment_available_seats_keys(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                assert set(seg["available_seats"].keys()) == FARE_CLASSES, (
                    f"{name}/{jid}/seg[{i}]: available_seats keys "
                    f"{set(seg['available_seats'].keys())} != {FARE_CLASSES}"
                )

    def test_journey_segment_available_seats_values(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                for fc, count in seg["available_seats"].items():
                    assert isinstance(count, int) and count >= 0, (
                        f"{name}/{jid}/seg[{i}]: available_seats['{fc}'] should be non-negative int, got {count}"
                    )

    def test_journey_segment_available_seat_types_keys(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                assert set(seg["available_seat_types"].keys()) == FARE_CLASSES, (
                    f"{name}/{jid}/seg[{i}]: available_seat_types keys "
                    f"{set(seg['available_seat_types'].keys())} != {FARE_CLASSES}"
                )

    def test_journey_segment_available_seat_types_values(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                for fc, types in seg["available_seat_types"].items():
                    assert isinstance(types, list), (
                        f"{name}/{jid}/seg[{i}]: available_seat_types['{fc}'] should be list, got {type(types)}"
                    )
                    for t in types:
                        assert t in SEAT_TYPES, (
                            f"{name}/{jid}/seg[{i}]: invalid seat type '{t}' in available_seat_types['{fc}']"
                        )

    def test_zero_seats_means_empty_seat_types(self, scenario_db):
        """If available_seats is 0 for a fare class, available_seat_types for that class must be an empty list."""
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                for fc in FARE_CLASSES:
                    if seg["available_seats"].get(fc, 0) == 0:
                        seat_types = seg["available_seat_types"].get(fc, [])
                        assert seat_types == [], (
                            f"{name}/{jid}/seg[{i}]: available_seats['{fc}'] "
                            f"is 0 but available_seat_types['{fc}'] is "
                            f"{seat_types} (expected [])"
                        )

    def test_journey_segment_fares_keys(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                assert set(seg["fares"].keys()) == FARE_CLASSES, (
                    f"{name}/{jid}/seg[{i}]: segment fares keys {set(seg['fares'].keys())} != {FARE_CLASSES}"
                )

    def test_journey_segment_fares_values(self, scenario_db):
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            for i, seg in enumerate(journey["segments"]):
                for fc, price in seg["fares"].items():
                    assert price is None or isinstance(price, (int, float)), (
                        f"{name}/{jid}/seg[{i}]: fares['{fc}'] should be float or None, got {type(price)}"
                    )

    def test_journey_origin_destination_chain(self, scenario_db):
        """For multi-segment journeys, segments should chain origin→destination."""
        name, data = scenario_db
        for jid, journey in data["journeys"].items():
            segs = journey["segments"]
            if len(segs) > 1:
                for k in range(len(segs) - 1):
                    assert segs[k]["destination"] == segs[k + 1]["origin"], (
                        f"{name}/{jid}: segment[{k}] destination "
                        f"'{segs[k]['destination']}' != segment[{k + 1}] "
                        f"origin '{segs[k + 1]['origin']}'"
                    )
            # First/last segment should match journey origin/destination
            assert segs[0]["origin"] == journey["origin"], (
                f"{name}/{jid}: first segment origin '{segs[0]['origin']}' != journey origin '{journey['origin']}'"
            )
            assert segs[-1]["destination"] == journey["destination"], (
                f"{name}/{jid}: last segment destination "
                f"'{segs[-1]['destination']}' != journey destination "
                f"'{journey['destination']}'"
            )


DISRUPTION_FIELDS = {
    "flight_number": str,
    "date": str,
    "disruption_type": str,
    "cause": str,
    "cause_category": str,
    "is_irrops": bool,
    "delay_minutes": (int, type(None)),
    "passenger_entitled_to": dict,
}

ENTITLEMENT_FIELDS = {
    "fee_waiver": bool,
    "refund_option": bool,
    "meal_voucher": bool,
    "hotel_accommodation": bool,
    "rebooking_window_days": int,
}


class TestDisruptions:
    def test_disruption_key_format(self, scenario_db):
        """Disruption keys should be '{flight_number}_{date}'."""
        name, data = scenario_db
        for key, dis in data["disruptions"].items():
            expected_key = f"{dis['flight_number']}_{dis['date']}"
            assert key == expected_key, f"{name}: disruption key '{key}' != expected '{expected_key}'"

    def test_disruption_fields(self, scenario_db):
        name, data = scenario_db
        for key, dis in data["disruptions"].items():
            for field, expected_type in DISRUPTION_FIELDS.items():
                assert field in dis, f"{name}/{key}: missing field '{field}'"
                assert isinstance(dis[field], expected_type), (
                    f"{name}/{key}: '{field}' should be {expected_type}, got {type(dis[field])}"
                )

    def test_disruption_date_format(self, scenario_db):
        name, data = scenario_db
        for key, dis in data["disruptions"].items():
            assert DATE_RE.match(dis["date"]), f"{name}/{key}: date '{dis['date']}' doesn't match YYYY-MM-DD"

    def test_disruption_type_enum(self, scenario_db):
        name, data = scenario_db
        for key, dis in data["disruptions"].items():
            assert dis["disruption_type"] in DISRUPTION_TYPES, (
                f"{name}/{key}: invalid disruption_type '{dis['disruption_type']}'"
            )

    def test_disruption_cause_category_enum(self, scenario_db):
        name, data = scenario_db
        for key, dis in data["disruptions"].items():
            assert dis["cause_category"] in CAUSE_CATEGORIES, (
                f"{name}/{key}: invalid cause_category '{dis['cause_category']}'"
            )

    def test_entitlement_fields(self, scenario_db):
        name, data = scenario_db
        for key, dis in data["disruptions"].items():
            ent = dis["passenger_entitled_to"]
            for field, expected_type in ENTITLEMENT_FIELDS.items():
                assert field in ent, f"{name}/{key}: passenger_entitled_to missing '{field}'"
                assert isinstance(ent[field], expected_type), (
                    f"{name}/{key}: passenger_entitled_to.{field} should be {expected_type}, got {type(ent[field])}"
                )


class TestReferentialIntegrity:
    def test_booking_journey_ids_exist_in_journeys(self, scenario_db):
        """Every journey_id referenced in a booking must exist in journeys."""
        name, data = scenario_db
        journey_ids = set(data["journeys"].keys())
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                assert bk["journey_id"] in journey_ids, (
                    f"{name}/{conf}/booking[{i}]: journey_id '{bk['journey_id']}' not in journeys table"
                )

    def test_booking_segment_flights_exist_in_journey(self, scenario_db):
        """Flight numbers in booking segments should appear in the referenced journey's segments."""
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                journey = data["journeys"].get(bk["journey_id"])
                if journey is None:
                    continue  # Caught by other test
                journey_flights = {s["flight_number"] for s in journey["segments"]}
                for j, seg in enumerate(bk["segments"]):
                    assert seg["flight_number"] in journey_flights, (
                        f"{name}/{conf}/booking[{i}]/seg[{j}]: "
                        f"flight '{seg['flight_number']}' not in "
                        f"journey '{bk['journey_id']}' segments"
                    )

    def test_booking_segments_have_own_journey_entry(self, scenario_db):
        """Verify booking segments have their own journey entry.

        Each booking segment (flight_number + date) should have a
        corresponding standalone journey entry in the journeys table
        (keyed as FL_{flight_number}_{YYYYMMDD}).
        """
        name, data = scenario_db
        for conf, res in data["reservations"].items():
            for i, bk in enumerate(res["bookings"]):
                for j, seg in enumerate(bk["segments"]):
                    date_compact = seg["date"].replace("-", "")
                    expected_key = f"FL_{seg['flight_number']}_{date_compact}"
                    assert expected_key in data["journeys"], (
                        f"{name}/{conf}/booking[{i}]/seg[{j}]: "
                        f"no standalone journey '{expected_key}' in "
                        f"journeys table for flight {seg['flight_number']} "
                        f"on {seg['date']}"
                    )

    def test_disruption_flight_exists_in_journeys(self, scenario_db):
        """Disrupted flights should reference a flight_number that appears somewhere in the journeys table."""
        name, data = scenario_db
        all_flights = set()
        for journey in data["journeys"].values():
            for seg in journey["segments"]:
                all_flights.add(seg["flight_number"])
        for key, dis in data["disruptions"].items():
            assert dis["flight_number"] in all_flights, (
                f"{name}/{key}: disrupted flight '{dis['flight_number']}' not found in any journey segment"
            )
