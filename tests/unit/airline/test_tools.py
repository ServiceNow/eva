"""Unit tests for airline tool functions."""

import copy

import pytest

from eva.assistant.tools.airline_tools import (
    add_baggage_allowance,
    add_meal_request,
    add_to_standby,
    assign_seat,
    cancel_reservation,
    get_disruption_info,
    get_flight_status,
    get_reservation,
    issue_hotel_voucher,
    issue_meal_voucher,
    issue_travel_credit,
    process_refund,
    rebook_flight,
    search_rebooking_options,
    transfer_to_agent,
)


@pytest.fixture
def sample_db():
    """Sample scenario database for testing (matches brainstorm DB schema).

    Scenario: John Doe's original flight SW100 was cancelled due to weather.
    He was rebooked onto SW200. The reservation has two segments:
      - SEG_SW100_20260320: cancelled (original flight)
      - SEG_SW200_20260320: confirmed (rebooked alternative)
    """
    base = {
        "reservations": {
            "ABC123": {
                "confirmation_number": "ABC123",
                "status": "changed",
                "passengers": [
                    {
                        "passenger_id": "PAX001",
                        "first_name": "John",
                        "last_name": "Doe",
                        "ticket_number": "0861234567890",
                        "email": "john.doe@gmail.com",
                        "phone": "+1-555-123-4567",
                        "elite_status": None,
                        "meal_preference": "none",
                        "seat_preference": "window",
                    }
                ],
                "bookings": [
                    {
                        "journey_id": "FL_SW100_20260320",
                        "fare_class": "main_cabin",
                        "fare_paid": 350.0,
                        "status": "cancelled",
                        "segments": [
                            {
                                "flight_number": "SW100",
                                "date": "2026-03-20",
                                "fare_paid": 350.0,
                                "seat": "15A",
                                "bags_checked": 1,
                                "meal_request": None,
                            }
                        ],
                    },
                    {
                        "journey_id": "FL_SW200_20260320",
                        "fare_class": "main_cabin",
                        "fare_paid": 400.0,
                        "status": "confirmed",
                        "segments": [
                            {
                                "flight_number": "SW200",
                                "date": "2026-03-20",
                                "fare_paid": 400.0,
                                "seat": "22C",
                                "bags_checked": 1,
                                "meal_request": None,
                            }
                        ],
                    },
                ],
                "booking_date": "2026-02-10T09:00:00-05:00",
                "fare_type": "non_refundable",
                "ancillaries": {"seat_selection_fee": 0, "bags_fee": 0},
            }
        },
        "journeys": {
            "FL_SW100_20260320": {
                "journey_id": "FL_SW100_20260320",
                "date": "2026-03-20",
                "origin": "LAX",
                "destination": "JFK",
                "num_stops": 0,
                "total_duration_minutes": 330,
                "fares": {
                    "basic_economy": None,
                    "main_cabin": None,
                    "premium_economy": None,
                    "business": None,
                    "first": None,
                },
                "segments": [
                    {
                        "segment_number": 1,
                        "flight_number": "SW100",
                        "origin": "LAX",
                        "destination": "JFK",
                        "scheduled_departure": "10:00",
                        "origin_utc_offset": -8,
                        "scheduled_arrival": "18:00",
                        "destination_utc_offset": -5,
                        "duration_minutes": 330,
                        "aircraft_type": "A321",
                        "status": "cancelled",
                        "delay_minutes": None,
                        "delay_reason": None,
                        "cancellation_reason": "weather",
                        "gate": "B12",
                        "available_seats": {
                            "basic_economy": 0,
                            "main_cabin": 0,
                            "premium_economy": 0,
                            "business": 0,
                            "first": 0,
                        },
                        "fares": {
                            "basic_economy": None,
                            "main_cabin": None,
                            "premium_economy": None,
                            "business": None,
                            "first": None,
                        },
                    }
                ],
                "status": "cancelled",
                "bookable": False,
            },
            "FL_SW200_20260320": {
                "journey_id": "FL_SW200_20260320",
                "date": "2026-03-20",
                "origin": "LAX",
                "destination": "JFK",
                "num_stops": 0,
                "total_duration_minutes": 340,
                "fares": {
                    "basic_economy": 280.0,
                    "main_cabin": 400.0,
                    "premium_economy": 600.0,
                    "business": 1300.0,
                    "first": 2600.0,
                },
                "segments": [
                    {
                        "segment_number": 1,
                        "flight_number": "SW200",
                        "origin": "LAX",
                        "destination": "JFK",
                        "scheduled_departure": "14:00",
                        "origin_utc_offset": -8,
                        "scheduled_arrival": "22:00",
                        "destination_utc_offset": -5,
                        "duration_minutes": 340,
                        "aircraft_type": "737-800",
                        "status": "scheduled",
                        "delay_minutes": None,
                        "delay_reason": None,
                        "cancellation_reason": None,
                        "gate": "A6",
                        "available_seats": {
                            "basic_economy": 12,
                            "main_cabin": 30,
                            "premium_economy": 15,
                            "business": 8,
                            "first": 2,
                        },
                        "fares": {
                            "basic_economy": 280.0,
                            "main_cabin": 400.0,
                            "premium_economy": 600.0,
                            "business": 1300.0,
                            "first": 2600.0,
                        },
                    }
                ],
                "status": "scheduled",
                "bookable": True,
            },
        },
        "disruptions": {
            "SW100_2026-03-20": {
                "flight_number": "SW100",
                "date": "2026-03-20",
                "disruption_type": "cancellation",
                "cause": "weather",
                "cause_category": "weather",
                "is_irrops": True,
                "delay_minutes": None,
                "passenger_entitled_to": {
                    "fee_waiver": True,
                    "refund_option": True,
                    "meal_voucher": True,
                    "hotel_accommodation": False,
                    "rebooking_window_days": 7,
                },
            }
        },
        "travel_credits": {},
        "meal_vouchers": {},
        "refunds": {},
        "_current_date": "2026-02-19",
    }

    # --- Multi-segment connecting journey: LAX→ORD→JFK ---
    # Journey definition with 2 segments (SW300 cancelled, SW400 scheduled)
    base["journeys"]["FL_SW300_SW400_20260322"] = {
        "journey_id": "FL_SW300_SW400_20260322",
        "date": "2026-03-22",
        "origin": "LAX",
        "destination": "JFK",
        "num_stops": 1,
        "total_duration_minutes": 420,
        "fares": {
            "basic_economy": None,
            "main_cabin": None,
            "premium_economy": None,
            "business": None,
            "first": None,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "SW300",
                "origin": "LAX",
                "destination": "ORD",
                "scheduled_departure": "08:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "14:00",
                "destination_utc_offset": -6,
                "duration_minutes": 240,
                "aircraft_type": "737-800",
                "status": "cancelled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": "mechanical",
                "gate": "C4",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 0,
                    "premium_economy": 0,
                    "business": 0,
                    "first": 0,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": None,
                    "premium_economy": None,
                    "business": None,
                    "first": None,
                },
            },
            {
                "segment_number": 2,
                "flight_number": "SW400",
                "origin": "ORD",
                "destination": "JFK",
                "scheduled_departure": "15:30",
                "origin_utc_offset": -6,
                "scheduled_arrival": "19:00",
                "destination_utc_offset": -5,
                "duration_minutes": 150,
                "aircraft_type": "A320",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "D8",
                "available_seats": {
                    "basic_economy": 5,
                    "main_cabin": 20,
                    "premium_economy": 10,
                    "business": 4,
                    "first": 1,
                },
                "fares": {
                    "basic_economy": 180.0,
                    "main_cabin": 250.0,
                    "premium_economy": 400.0,
                    "business": 900.0,
                    "first": 1800.0,
                },
            },
        ],
        "status": "cancelled",
        "bookable": False,
    }

    # Replacement journey for the LAX→ORD leg
    base["journeys"]["FL_SW350_20260322"] = {
        "journey_id": "FL_SW350_20260322",
        "date": "2026-03-22",
        "origin": "LAX",
        "destination": "ORD",
        "num_stops": 0,
        "total_duration_minutes": 250,
        "fares": {
            "basic_economy": 150.0,
            "main_cabin": 220.0,
            "premium_economy": 350.0,
            "business": 800.0,
            "first": 1600.0,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "SW350",
                "origin": "LAX",
                "destination": "ORD",
                "scheduled_departure": "09:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "15:10",
                "destination_utc_offset": -6,
                "duration_minutes": 250,
                "aircraft_type": "737-800",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "C7",
                "available_seats": {
                    "basic_economy": 10,
                    "main_cabin": 25,
                    "premium_economy": 8,
                    "business": 6,
                    "first": 2,
                },
                "fares": {
                    "basic_economy": 150.0,
                    "main_cabin": 220.0,
                    "premium_economy": 350.0,
                    "business": 800.0,
                    "first": 1600.0,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }

    # Booking entry in reservation ABC123 for the connecting journey
    base["reservations"]["ABC123"]["bookings"].append(
        {
            "journey_id": "FL_SW300_SW400_20260322",
            "fare_class": "main_cabin",
            "fare_paid": 450.0,
            "status": "confirmed",
            "segments": [
                {
                    "flight_number": "SW300",
                    "date": "2026-03-22",
                    "fare_paid": 200.0,
                    "seat": "10A",
                    "bags_checked": 1,
                    "meal_request": "vegetarian",
                },
                {
                    "flight_number": "SW400",
                    "date": "2026-03-22",
                    "fare_paid": 250.0,
                    "seat": "12C",
                    "bags_checked": 1,
                    "meal_request": "vegetarian",
                },
            ],
        }
    )

    # Disruption for SW300
    base["disruptions"]["SW300_2026-03-22"] = {
        "flight_number": "SW300",
        "date": "2026-03-22",
        "disruption_type": "cancellation",
        "cause": "mechanical",
        "cause_category": "airline_fault",
        "is_irrops": True,
        "delay_minutes": None,
        "passenger_entitled_to": {
            "fee_waiver": True,
            "refund_option": True,
            "meal_voucher": True,
            "hotel_accommodation": False,
            "rebooking_window_days": 7,
        },
    }

    # ============================================
    # CONF01 — Standard nonstop, voluntary rebooking (LAX→JFK, 2026-03-27)
    # Dates differ from existing ABC123 (2026-03-20) to avoid search-count conflicts.
    # ============================================
    base["reservations"]["CONF01"] = {
        "confirmation_number": "CONF01",
        "status": "confirmed",
        "passengers": [
            {
                "passenger_id": "PAX001",
                "first_name": "John",
                "last_name": "Smith",
                "ticket_number": "0861111111111",
                "email": "john.smith@gmail.com",
                "phone": "+1-555-200-0001",
                "elite_status": None,
                "meal_preference": "none",
                "seat_preference": "window",
            }
        ],
        "bookings": [
            {
                "journey_id": "FL_AA100_20260327",
                "fare_class": "main_cabin",
                "fare_paid": 350.0,
                "status": "confirmed",
                "segments": [
                    {
                        "flight_number": "AA100",
                        "date": "2026-03-27",
                        "fare_paid": 350.0,
                        "seat": "15A",
                        "bags_checked": 1,
                        "meal_request": None,
                    }
                ],
            }
        ],
        "booking_date": "2026-02-01T09:00:00-05:00",
        "fare_type": "non_refundable",
        "ancillaries": {"seat_selection_fee": 0, "bags_fee": 0},
    }
    base["journeys"]["FL_AA100_20260327"] = {
        "journey_id": "FL_AA100_20260327",
        "date": "2026-03-27",
        "origin": "LAX",
        "destination": "JFK",
        "num_stops": 0,
        "total_duration_minutes": 330,
        "fares": {
            "basic_economy": None,
            "main_cabin": 350.0,
            "premium_economy": 500.0,
            "business": 1000.0,
            "first": 2000.0,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "AA100",
                "origin": "LAX",
                "destination": "JFK",
                "scheduled_departure": "08:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "16:00",
                "destination_utc_offset": -5,
                "duration_minutes": 330,
                "aircraft_type": "A321",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "A1",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 25,
                    "premium_economy": 10,
                    "business": 4,
                    "first": 2,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": 350.0,
                    "premium_economy": 500.0,
                    "business": 1000.0,
                    "first": 2000.0,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }
    base["journeys"]["FL_AA200_20260327"] = {
        "journey_id": "FL_AA200_20260327",
        "date": "2026-03-27",
        "origin": "LAX",
        "destination": "JFK",
        "num_stops": 0,
        "total_duration_minutes": 335,
        "fares": {
            "basic_economy": None,  # No basic_economy fare — used by multi-search test
            "main_cabin": 400.0,
            "premium_economy": 600.0,
            "business": 1200.0,
            "first": 2400.0,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "AA200",
                "origin": "LAX",
                "destination": "JFK",
                "scheduled_departure": "14:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "22:05",
                "destination_utc_offset": -5,
                "duration_minutes": 335,
                "aircraft_type": "737-800",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "A5",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 25,
                    "premium_economy": 12,
                    "business": 6,
                    "first": 2,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": 400.0,
                    "premium_economy": 600.0,
                    "business": 1200.0,
                    "first": 2400.0,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }

    # ============================================
    # CONF02 — IRROPS cancellation (SFO→ORD, 2026-03-29)
    # ============================================
    base["reservations"]["CONF02"] = {
        "confirmation_number": "CONF02",
        "status": "confirmed",
        "passengers": [
            {
                "passenger_id": "PAX001",
                "first_name": "Jane",
                "last_name": "Doe",
                "ticket_number": "0862222222222",
                "email": "jane.doe@gmail.com",
                "phone": "+1-555-200-0002",
                "elite_status": None,
                "meal_preference": "none",
                "seat_preference": "aisle",
            }
        ],
        "bookings": [
            {
                "journey_id": "FL_BB100_20260329",
                "fare_class": "main_cabin",
                "fare_paid": 280.0,
                "status": "confirmed",
                "segments": [
                    {
                        "flight_number": "BB100",
                        "date": "2026-03-29",
                        "fare_paid": 280.0,
                        "seat": "10C",
                        "bags_checked": 1,
                        "meal_request": None,
                    }
                ],
            }
        ],
        "booking_date": "2026-02-05T10:00:00-05:00",
        "fare_type": "non_refundable",
        "ancillaries": {"seat_selection_fee": 0, "bags_fee": 0},
    }
    base["journeys"]["FL_BB100_20260329"] = {
        "journey_id": "FL_BB100_20260329",
        "date": "2026-03-29",
        "origin": "SFO",
        "destination": "ORD",
        "num_stops": 0,
        "total_duration_minutes": 270,
        "fares": {
            "basic_economy": None,
            "main_cabin": None,
            "premium_economy": None,
            "business": None,
            "first": None,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "BB100",
                "origin": "SFO",
                "destination": "ORD",
                "scheduled_departure": "07:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "13:00",
                "destination_utc_offset": -6,
                "duration_minutes": 270,
                "aircraft_type": "A320",
                "status": "cancelled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": "operational",
                "gate": "B5",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 0,
                    "premium_economy": 0,
                    "business": 0,
                    "first": 0,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": None,
                    "premium_economy": None,
                    "business": None,
                    "first": None,
                },
            }
        ],
        "status": "cancelled",
        "bookable": False,
    }
    base["journeys"]["FL_BB200_20260329"] = {
        "journey_id": "FL_BB200_20260329",
        "date": "2026-03-29",
        "origin": "SFO",
        "destination": "ORD",
        "num_stops": 0,
        "total_duration_minutes": 280,
        "fares": {
            "basic_economy": 200.0,
            "main_cabin": 300.0,
            "premium_economy": 450.0,
            "business": 900.0,
            "first": 1800.0,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "BB200",
                "origin": "SFO",
                "destination": "ORD",
                "scheduled_departure": "12:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "18:40",
                "destination_utc_offset": -6,
                "duration_minutes": 280,
                "aircraft_type": "737-800",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "C3",
                "available_seats": {
                    "basic_economy": 10,
                    "main_cabin": 20,
                    "premium_economy": 8,
                    "business": 4,
                    "first": 2,
                },
                "fares": {
                    "basic_economy": 200.0,
                    "main_cabin": 300.0,
                    "premium_economy": 450.0,
                    "business": 900.0,
                    "first": 1800.0,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }
    base["disruptions"]["BB100_2026-03-29"] = {
        "flight_number": "BB100",
        "date": "2026-03-29",
        "disruption_type": "cancellation",
        "cause": "operational",
        "cause_category": "airline_fault",
        "is_irrops": True,
        "delay_minutes": None,
        "passenger_entitled_to": {
            "fee_waiver": True,
            "refund_option": True,
            "meal_voucher": True,
            "hotel_accommodation": False,
            "rebooking_window_days": 7,
        },
    }

    # ============================================
    # CONF03 — Multi-segment connecting (LAX→ORD→JFK, 2026-03-25), partial rebook
    # ============================================
    base["reservations"]["CONF03"] = {
        "confirmation_number": "CONF03",
        "status": "confirmed",
        "passengers": [
            {
                "passenger_id": "PAX001",
                "first_name": "Bob",
                "last_name": "Johnson",
                "ticket_number": "0863333333333",
                "email": "bob.johnson@gmail.com",
                "phone": "+1-555-200-0003",
                "elite_status": None,
                "meal_preference": "vegetarian",
                "seat_preference": "window",
            }
        ],
        "bookings": [
            {
                "journey_id": "FL_CC300_CC400_20260325",
                "fare_class": "main_cabin",
                "fare_paid": 450.0,
                "status": "confirmed",
                "segments": [
                    {
                        "flight_number": "CC300",
                        "date": "2026-03-25",
                        "fare_paid": 200.0,
                        "seat": "10A",
                        "bags_checked": 1,
                        "meal_request": "vegetarian",
                    },
                    {
                        "flight_number": "CC400",
                        "date": "2026-03-25",
                        "fare_paid": 250.0,
                        "seat": "12C",
                        "bags_checked": 1,
                        "meal_request": "vegetarian",
                    },
                ],
            }
        ],
        "booking_date": "2026-02-08T11:00:00-05:00",
        "fare_type": "non_refundable",
        "ancillaries": {"seat_selection_fee": 0, "bags_fee": 0},
    }
    base["journeys"]["FL_CC300_CC400_20260325"] = {
        "journey_id": "FL_CC300_CC400_20260325",
        "date": "2026-03-25",
        "origin": "LAX",
        "destination": "JFK",
        "num_stops": 1,
        "total_duration_minutes": 420,
        "fares": {
            "basic_economy": None,
            "main_cabin": None,
            "premium_economy": None,
            "business": None,
            "first": None,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "CC300",
                "origin": "LAX",
                "destination": "ORD",
                "scheduled_departure": "07:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "13:00",
                "destination_utc_offset": -6,
                "duration_minutes": 240,
                "aircraft_type": "A321",
                "status": "cancelled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": "mechanical",
                "gate": "D1",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 0,
                    "premium_economy": 0,
                    "business": 0,
                    "first": 0,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": None,
                    "premium_economy": None,
                    "business": None,
                    "first": None,
                },
            },
            {
                "segment_number": 2,
                "flight_number": "CC400",
                "origin": "ORD",
                "destination": "JFK",
                "scheduled_departure": "15:00",
                "origin_utc_offset": -6,
                "scheduled_arrival": "18:30",
                "destination_utc_offset": -5,
                "duration_minutes": 150,
                "aircraft_type": "737-800",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "E2",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 0,
                    "premium_economy": 0,
                    "business": 0,
                    "first": 0,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": None,
                    "premium_economy": None,
                    "business": None,
                    "first": None,
                },
            },
        ],
        "status": "cancelled",
        "bookable": False,
    }
    base["journeys"]["FL_CC350_20260325"] = {
        "journey_id": "FL_CC350_20260325",
        "date": "2026-03-25",
        "origin": "LAX",
        "destination": "ORD",
        "num_stops": 0,
        "total_duration_minutes": 250,
        "fares": {
            "basic_economy": 150.0,
            "main_cabin": 220.0,
            "premium_economy": 350.0,
            "business": 700.0,
            "first": 1400.0,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "CC350",
                "origin": "LAX",
                "destination": "ORD",
                "scheduled_departure": "09:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "15:10",
                "destination_utc_offset": -6,
                "duration_minutes": 250,
                "aircraft_type": "737-800",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "D7",
                "available_seats": {
                    "basic_economy": 10,
                    "main_cabin": 20,
                    "premium_economy": 8,
                    "business": 4,
                    "first": 2,
                },
                "fares": {
                    "basic_economy": 150.0,
                    "main_cabin": 220.0,
                    "premium_economy": 350.0,
                    "business": 700.0,
                    "first": 1400.0,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }
    base["disruptions"]["CC300_2026-03-25"] = {
        "flight_number": "CC300",
        "date": "2026-03-25",
        "disruption_type": "cancellation",
        "cause": "mechanical",
        "cause_category": "airline_fault",
        "is_irrops": True,
        "delay_minutes": None,
        "passenger_entitled_to": {
            "fee_waiver": True,
            "refund_option": True,
            "meal_voucher": True,
            "hotel_accommodation": True,
            "rebooking_window_days": 7,
        },
    }

    # ============================================
    # CONF04 — Round-trip, non-refundable (cancel flows)
    # ============================================
    base["reservations"]["CONF04"] = {
        "confirmation_number": "CONF04",
        "status": "confirmed",
        "passengers": [
            {
                "passenger_id": "PAX001",
                "first_name": "Alice",
                "last_name": "Williams",
                "ticket_number": "0864444444444",
                "email": "alice.williams@gmail.com",
                "phone": "+1-555-200-0004",
                "elite_status": None,
                "meal_preference": "none",
                "seat_preference": "window",
            }
        ],
        "bookings": [
            {
                "journey_id": "FL_DD100_20260401",
                "fare_class": "main_cabin",
                "fare_paid": 350.0,
                "status": "confirmed",
                "segments": [
                    {
                        "flight_number": "DD100",
                        "date": "2026-04-01",
                        "fare_paid": 350.0,
                        "seat": "18A",
                        "bags_checked": 1,
                        "meal_request": None,
                    }
                ],
            },
            {
                "journey_id": "FL_DD200_20260415",
                "fare_class": "main_cabin",
                "fare_paid": 350.0,
                "status": "confirmed",
                "segments": [
                    {
                        "flight_number": "DD200",
                        "date": "2026-04-15",
                        "fare_paid": 350.0,
                        "seat": "18A",
                        "bags_checked": 1,
                        "meal_request": None,
                    }
                ],
            },
        ],
        "booking_date": "2026-02-10T12:00:00-05:00",
        "fare_type": "non_refundable",
        "ancillaries": {"seat_selection_fee": 0, "bags_fee": 0},
    }
    base["journeys"]["FL_DD100_20260401"] = {
        "journey_id": "FL_DD100_20260401",
        "date": "2026-04-01",
        "origin": "DCA",
        "destination": "LAX",
        "num_stops": 0,
        "total_duration_minutes": 330,
        "fares": {
            "basic_economy": None,
            "main_cabin": 350.0,
            "premium_economy": 550.0,
            "business": 1100.0,
            "first": 2200.0,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "DD100",
                "origin": "DCA",
                "destination": "LAX",
                "scheduled_departure": "08:00",
                "origin_utc_offset": -5,
                "scheduled_arrival": "11:30",
                "destination_utc_offset": -8,
                "duration_minutes": 330,
                "aircraft_type": "A321",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "F1",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 20,
                    "premium_economy": 8,
                    "business": 4,
                    "first": 2,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": 350.0,
                    "premium_economy": 550.0,
                    "business": 1100.0,
                    "first": 2200.0,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }
    base["journeys"]["FL_DD200_20260415"] = {
        "journey_id": "FL_DD200_20260415",
        "date": "2026-04-15",
        "origin": "LAX",
        "destination": "DCA",
        "num_stops": 0,
        "total_duration_minutes": 330,
        "fares": {
            "basic_economy": None,
            "main_cabin": 350.0,
            "premium_economy": 550.0,
            "business": 1100.0,
            "first": 2200.0,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "DD200",
                "origin": "LAX",
                "destination": "DCA",
                "scheduled_departure": "13:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "21:30",
                "destination_utc_offset": -5,
                "duration_minutes": 330,
                "aircraft_type": "A321",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "G2",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 20,
                    "premium_economy": 8,
                    "business": 4,
                    "first": 2,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": 350.0,
                    "premium_economy": 550.0,
                    "business": 1100.0,
                    "first": 2200.0,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }

    return base


def test_get_reservation_success(sample_db):
    """Test successful reservation lookup."""
    params = {"confirmation_number": "abc123", "last_name": "Doe"}
    result = get_reservation(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["reservation"]["confirmation_number"] == "ABC123"
    assert len(result["reservation"]["bookings"]) == 3
    assert result["reservation"]["bookings"][0]["status"] == "cancelled"
    assert result["reservation"]["bookings"][1]["status"] == "confirmed"


def test_get_reservation_writes_session(sample_db):
    """Successful get_reservation should write confirmation_number and last_name to db session."""
    params = {"confirmation_number": "abc123", "last_name": "Doe"}
    get_reservation(params, sample_db, call_index=1)

    assert sample_db["session"]["confirmation_number"] == "ABC123"
    assert sample_db["session"]["last_name"] == "doe"


def test_get_reservation_session_last_name_lowercased(sample_db):
    """Session last_name should be stored lowercase regardless of input case."""
    params = {"confirmation_number": "ABC123", "last_name": "DOE"}
    get_reservation(params, sample_db, call_index=1)

    assert sample_db["session"]["last_name"] == "doe"


def test_get_reservation_failed_auth_does_not_write_session(sample_db):
    """Failed authentication (bad last name) should not write to db session."""
    params = {"confirmation_number": "ABC123", "last_name": "Wrong"}
    result = get_reservation(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert "session" not in sample_db


def test_get_reservation_not_found_does_not_write_session(sample_db):
    """Reservation not found should not write to db session."""
    params = {"confirmation_number": "XXXXXX", "last_name": "Smith"}
    result = get_reservation(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "not_found"
    assert "XXXXXX" in result["message"]
    assert "session" not in sample_db


def test_get_reservation_enriches_with_flight_details(sample_db):
    """Bookings returned from get_reservation are enriched with flight details from the journeys table."""
    params = {"confirmation_number": "ABC123", "last_name": "Doe"}
    result = get_reservation(params, sample_db, call_index=1)

    assert result["status"] == "success"
    bookings_by_journey = {b["journey_id"]: b for b in result["reservation"]["bookings"]}

    cancelled = bookings_by_journey["FL_SW100_20260320"]
    cancelled_seg = cancelled["segments"][0]
    assert cancelled_seg["origin"] == "LAX"
    assert cancelled_seg["destination"] == "JFK"
    assert cancelled_seg["scheduled_departure"] == "10:00"
    assert cancelled_seg["scheduled_arrival"] == "18:00"

    confirmed = bookings_by_journey["FL_SW200_20260320"]
    confirmed_seg = confirmed["segments"][0]
    assert confirmed_seg["origin"] == "LAX"
    assert confirmed_seg["destination"] == "JFK"
    assert confirmed_seg["scheduled_departure"] == "14:00"
    assert confirmed_seg["scheduled_arrival"] == "22:00"

    # Enrichment must not mutate the underlying DB
    db_seg = sample_db["reservations"]["ABC123"]["bookings"][1]["segments"][0]
    assert "scheduled_departure" not in db_seg


def test_get_flight_status_success(sample_db):
    """Test successful flight status lookup."""
    params = {"flight_number": "SW200", "flight_date": "2026-03-20"}
    result = get_flight_status(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["journey"]["segments"][0]["flight_number"] == "SW200"
    assert result["journey"]["status"] == "scheduled"


def test_get_flight_status_cancelled(sample_db):
    """Test flight status lookup for cancelled flight."""
    params = {"flight_number": "SW100", "flight_date": "2026-03-20"}
    result = get_flight_status(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["journey"]["segments"][0]["flight_number"] == "SW100"
    assert result["journey"]["status"] == "cancelled"
    assert result["journey"]["segments"][0]["cancellation_reason"] == "weather"


def test_get_flight_status_not_found(sample_db):
    """Test flight not found."""
    params = {"flight_number": "SW999", "flight_date": "2026-03-20"}
    result = get_flight_status(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "not_found"


def test_get_disruption_info_success(sample_db):
    """Test successful disruption info lookup by flight number."""
    params = {"flight_number": "SW100", "date": "2026-03-20"}
    result = get_disruption_info(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["disruption"]["disruption_type"] == "cancellation"
    assert result["disruption"]["cause"] == "weather"
    assert result["disruption"]["is_irrops"] is True
    assert result["disruption"]["passenger_entitled_to"]["fee_waiver"] is True


def test_search_rebooking_options_success(sample_db):
    """Test successful flight search (only bookable flights returned)."""
    params = {
        "origin": "LAX",
        "destination": "JFK",
        "date": "2026-03-20",
        "passenger_count": 1,
        "fare_class": "main_cabin",
    }
    result = search_rebooking_options(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["count"] == 1
    assert len(result["options"]) == 1
    assert result["options"][0]["departure_time"] == "14:00"

    # Verify available_seat_types is NOT in the search response (removed in V2)
    option = result["options"][0]
    assert "available_seat_types" not in option


def test_search_rebooking_options_no_results(sample_db):
    """Test flight search with no results."""
    params = {
        "origin": "LAX",
        "destination": "SFO",  # No flights to SFO
        "date": "2026-03-20",
        "passenger_count": 1,
        "fare_class": "main_cabin",
    }
    result = search_rebooking_options(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["count"] == 0
    assert len(result["options"]) == 0


def test_search_rebooking_options_cabin_any(sample_db):
    """Test search with cabin_class='any'."""
    params = {
        "origin": "LAX",
        "destination": "JFK",
        "date": "2026-03-20",
        "passenger_count": 1,
        "fare_class": "any",
    }
    result = search_rebooking_options(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["count"] == 1


def test_rebook_flight_voluntary_success(sample_db):
    """Test voluntary rebooking with fee (rebooking cancelled segment to same flight)."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "voluntary",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["confirmation_number"] == "ABC123"
    assert result["cost_summary"]["change_fee"] == 75  # main_cabin fee
    assert result["cost_summary"]["fare_difference"] == 50  # 400 - 350
    assert result["cost_summary"]["total_collected"] == 125  # 75 + 50

    # Check mutations: original 3 booking journeys + 1 new appended
    reservation = db["reservations"]["ABC123"]
    assert reservation["status"] == "changed"
    assert len(reservation["bookings"]) == 4
    assert reservation["bookings"][0]["status"] == "cancelled"
    new_booking = reservation["bookings"][3]
    assert new_booking["status"] == "confirmed"
    assert new_booking["journey_id"] == "FL_SW200_20260320"
    assert len(new_booking["segments"]) == 1
    assert new_booking["segments"][0]["flight_number"] == "SW200"
    assert new_booking["segments"][0]["seat"] is None


def test_rebook_flight_irrops_no_fees(sample_db):
    """Test IRROPS rebooking with no fees."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "irrops_cancellation",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 0
    assert result["cost_summary"]["total_collected"] == 0
    assert result["cost_summary"]["fee_waived"] is True


def test_rebook_flight_voluntary_basic_economy_fee(sample_db):
    """Voluntary rebook on a Basic Economy booking: flat $75 (not the same-day $199 penalty)."""
    db = copy.deepcopy(sample_db)
    # Flip the cancelled booking's fare_class to basic_economy for this test.
    db["reservations"]["ABC123"]["bookings"][0]["fare_class"] = "basic_economy"
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "voluntary",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 75


def test_rebook_flight_same_day_basic_economy_penalty_fee(sample_db):
    """Same-day rebook on a Basic Economy booking: $199 penalty."""
    db = copy.deepcopy(sample_db)
    db["reservations"]["ABC123"]["bookings"][0]["fare_class"] = "basic_economy"
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "same_day",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 199


def test_rebook_flight_same_day_main_cabin_standard_fee(sample_db):
    """Same-day rebook on Main Cabin: standard $75 fee (no penalty)."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "same_day",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 75


def test_rebook_flight_same_day_business_class_fee(sample_db):
    """Same-day rebook on Business Class: $75 (Business pays for same-day, unlike voluntary)."""
    db = copy.deepcopy(sample_db)
    db["reservations"]["ABC123"]["bookings"][0]["fare_class"] = "business"
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "same_day",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 75


def test_rebook_flight_voluntary_business_class_no_fee(sample_db):
    """Voluntary rebook on Business Class: $0 (free for Business/First in advance changes)."""
    db = copy.deepcopy(sample_db)
    db["reservations"]["ABC123"]["bookings"][0]["fare_class"] = "business"
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "voluntary",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 0


def test_rebook_flight_same_day_basic_economy_waived_for_elite(sample_db):
    """Gold/Platinum waiver zeros the same-day penalty for Basic Economy."""
    db = copy.deepcopy(sample_db)
    db["reservations"]["ABC123"]["bookings"][0]["fare_class"] = "basic_economy"
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "same_day",
        "waive_change_fee": True,
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 0
    assert result["cost_summary"]["fee_waived"] is True


def test_rebook_flight_reservation_not_found(sample_db):
    """Test rebooking with invalid reservation."""
    params = {
        "confirmation_number": "XXXXXX",
        "journey_id": "FL_SW100_20260320",
        "new_journey_id": "FL_SW200_20260320",
        "rebooking_type": "voluntary",
        "waive_change_fee": False,
    }
    result = rebook_flight(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "not_found"


def test_assign_seat_success(sample_db):
    """Test successful seat assignment on confirmed segment."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "passenger_id": "PAX001",
        "journey_id": "FL_SW200_20260320",
        "seat_preference": "window",
    }
    result = assign_seat(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["seat_assigned"] == "21A"  # base_row=20 (default) + passenger_index=1, letter=A
    assert result["preference"] == "window"

    # Check mutation on the confirmed booking's flight segment
    reservation = db["reservations"]["ABC123"]
    assert reservation["bookings"][1]["segments"][0]["seat"] == "21A"


def test_add_baggage_allowance_success(sample_db):
    """Test adding baggage allowance on confirmed segment."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW200_20260320",
        "num_bags": 2,
    }
    result = add_baggage_allowance(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["bags_checked"] == 2

    # Check mutation on the confirmed booking's flight segment
    reservation = db["reservations"]["ABC123"]
    assert reservation["bookings"][1]["segments"][0]["bags_checked"] == 2


def test_add_baggage_invalid_count(sample_db):
    """Test invalid baggage count."""
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW200_20260320",
        "num_bags": 10,  # Exceeds maximum
    }
    result = add_baggage_allowance(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "invalid_bag_count"


def test_add_meal_request_success(sample_db):
    """Test adding meal request on confirmed segment."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "passenger_id": "PAX001",
        "journey_id": "FL_SW200_20260320",
        "meal_type": "vegetarian",
    }
    result = add_meal_request(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["meal_type"] == "vegetarian"

    # Check mutation on the confirmed booking's flight segment
    reservation = db["reservations"]["ABC123"]
    assert reservation["bookings"][1]["segments"][0]["meal_request"] == "vegetarian"


def test_issue_travel_credit_success(sample_db):
    """Test issuing travel credit."""
    params = {
        "confirmation_number": "ABC123",
        "passenger_id": "PAX001",
        "amount": 50.0,
        "credit_reason": "service_recovery",
    }
    result = issue_travel_credit(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["amount"] == 50.0
    assert result["credit_code"] == "TCABC123PAX"
    assert result["valid_months"] == 12


def test_issue_travel_credit_invalid_reason(sample_db):
    """Test credit with invalid reason."""
    params = {
        "confirmation_number": "ABC123",
        "passenger_id": "PAX001",
        "amount": 50.0,
        "credit_reason": "invalid_reason",
    }
    result = issue_travel_credit(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "invalid_credit_reason"


def test_issue_hotel_voucher_success(sample_db):
    """Test issuing hotel voucher."""
    params = {
        "confirmation_number": "ABC123",
        "passenger_id": "PAX001",
        "num_nights": 2,
    }
    result = issue_hotel_voucher(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["number_of_nights"] == 2
    assert result["voucher_code"] == "HOTEL-ABC123"


def test_issue_meal_voucher_success(sample_db):
    """Test issuing meal voucher."""
    params = {
        "confirmation_number": "ABC123",
        "passenger_id": "PAX001",
        "voucher_reason": "delay_over_4_hours",
    }
    result = issue_meal_voucher(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["amount"] == 15
    assert "MEAL-ABC123" in result["voucher_code"]


def test_cancel_segment_refundable(sample_db):
    """Test cancelling a segment on a refundable reservation."""
    db = copy.deepcopy(sample_db)
    db["reservations"]["ABC123"]["fare_type"] = "refundable"
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW200_20260320",
        "cancellation_reason": "voluntary",
    }
    result = cancel_reservation(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["journey_id"] == "FL_SW200_20260320"
    assert result["is_refundable"] is True
    assert result["cancellation_fee"] == 0
    assert result["refund_amount_eligible"] == 400.0
    assert result["credit_amount_eligible"] == 0

    # Not all journeys cancelled (connecting journey still confirmed) → reservation still active
    assert result["reservation_status"] == "active"


def test_cancel_segment_non_refundable(sample_db):
    """Test cancelling a segment on a non-refundable reservation."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW200_20260320",
        "cancellation_reason": "voluntary",
    }
    result = cancel_reservation(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["is_refundable"] is False
    assert result["cancellation_fee"] == 100
    assert result["refund_amount_eligible"] == 0
    assert result["credit_amount_eligible"] == 300.0  # 400 - 100


def test_cancel_segment_already_cancelled(sample_db):
    """Test cancelling an already-cancelled segment."""
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "cancellation_reason": "voluntary",
    }
    result = cancel_reservation(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "already_cancelled"


def test_process_refund_success(sample_db):
    """Test processing refund."""
    params = {
        "confirmation_number": "ABC123",
        "refund_amount": 350.0,
        "refund_type": "full_fare",
    }
    result = process_refund(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["refund_amount"] == 350.0
    assert result["refund_id"] == "REF-ABC123-001"
    assert result["processing_days"] == 7


def test_transfer_to_agent_success(sample_db):
    """Test transferring to live agent."""
    params = {
        "confirmation_number": "ABC123",
        "transfer_reason": "passenger_requested",
        "issue_summary": "Customer wants to speak to manager",
    }
    result = transfer_to_agent(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["transfer_id"] == "TRF-ABC123-001"
    assert result["transfer_reason"] == "passenger_requested"
    assert "Transferring" in result["message"]


def test_add_to_standby_success(sample_db):
    """Test adding passengers to standby."""
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW200_20260320",
        "passenger_ids": ["PAX001"],
    }
    result = add_to_standby(params, sample_db, call_index=1)

    assert result["status"] == "success"
    assert result["journey_id"] == "FL_SW200_20260320"
    assert result["standby_list_position"] == 1


def test_add_to_standby_cancelled_flight(sample_db):
    """Test adding to standby for cancelled flight."""
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW100_20260320",
        "passenger_ids": ["PAX001"],
    }
    result = add_to_standby(params, sample_db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "flight_cancelled"


def test_rebook_flight_partial_leg_success(sample_db):
    """Test partial rebook of a single cancelled leg in a multi-segment journey."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW300_SW400_20260322",
        "new_journey_id": "FL_SW350_20260322",
        "rebooking_type": "voluntary",
        "waive_change_fee": False,
        "flight_number": "SW300",
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["partial_rebook"] is True

    # Cost summary uses single-leg fare comparison: $220 (new) - $200 (old SW300) = $20
    assert result["cost_summary"]["fare_difference"] == 20.0
    assert result["cost_summary"]["change_fee"] == 75  # main_cabin voluntary
    assert result["cost_summary"]["total_collected"] == 95.0  # 75 + 20

    # Replaced segment info
    assert result["replaced_segment"]["flight_number"] == "SW300"
    assert result["replaced_segment"]["fare_paid"] == 200.0

    # Kept segments info
    assert len(result["kept_segments"]) == 1
    assert result["kept_segments"][0]["flight_number"] == "SW400"
    assert result["kept_segments"][0]["fare_paid"] == 250.0
    assert result["kept_segments"][0]["seat"] == "12C"

    # Check mutations on reservation
    reservation = db["reservations"]["ABC123"]

    # Old booking cancelled
    old_booking = [j for j in reservation["bookings"] if j["journey_id"] == "FL_SW300_SW400_20260322"]
    assert old_booking[0]["status"] == "cancelled"

    # New replacement booking created
    replacement = [
        j for j in reservation["bookings"] if j["journey_id"] == "FL_SW350_20260322" and j["status"] == "confirmed"
    ]
    assert len(replacement) == 1
    assert replacement[0]["segments"][0]["flight_number"] == "SW350"
    assert replacement[0]["segments"][0]["fare_paid"] == 220.0

    # Kept booking created preserving ancillaries
    kept = [
        j
        for j in reservation["bookings"]
        if j["journey_id"] == "FL_SW300_SW400_20260322" and j["status"] == "confirmed"
    ]
    assert len(kept) == 1
    assert len(kept[0]["segments"]) == 1
    kept_seg = kept[0]["segments"][0]
    assert kept_seg["flight_number"] == "SW400"
    assert kept_seg["fare_paid"] == 250.0
    assert kept_seg["seat"] == "12C"
    assert kept_seg["bags_checked"] == 1
    assert kept_seg["meal_request"] == "vegetarian"


def test_rebook_flight_partial_leg_irrops(sample_db):
    """Test partial rebook under IRROPS — no fees or fare difference collected."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW300_SW400_20260322",
        "new_journey_id": "FL_SW350_20260322",
        "rebooking_type": "irrops_cancellation",
        "waive_change_fee": False,
        "flight_number": "SW300",
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "success"
    assert result["partial_rebook"] is True
    assert result["cost_summary"]["change_fee"] == 0
    assert result["cost_summary"]["total_collected"] == 0
    assert result["cost_summary"]["fee_waived"] is True
    # fare_difference still computed for informational purposes
    assert result["cost_summary"]["fare_difference"] == 20.0


def test_rebook_flight_partial_flight_not_found(sample_db):
    """Test partial rebook with an invalid flight_number."""
    db = copy.deepcopy(sample_db)
    params = {
        "confirmation_number": "ABC123",
        "journey_id": "FL_SW300_SW400_20260322",
        "new_journey_id": "FL_SW350_20260322",
        "rebooking_type": "voluntary",
        "waive_change_fee": False,
        "flight_number": "SW999",
    }
    result = rebook_flight(params, db, call_index=1)

    assert result["status"] == "error"
    assert result["error_type"] == "flight_not_found"
    assert "SW999" in result["message"]


@pytest.fixture
def fresh_db(sample_db):
    """Deep copy of sample_db so mutations in one test don't affect others."""
    return copy.deepcopy(sample_db)


def test_flow_standard_voluntary_rebooking(fresh_db):
    """CONF01: lookup → search → rebook → assign seat → add baggage."""
    db = fresh_db

    # Step 1: get reservation
    result = get_reservation({"confirmation_number": "CONF01", "last_name": "Smith"}, db, call_index=1)
    assert result["status"] == "success"
    assert result["reservation"]["confirmation_number"] == "CONF01"

    # Step 2: search for alternatives (LAX→JFK on 2026-03-27)
    result = search_rebooking_options(
        {"origin": "LAX", "destination": "JFK", "date": "2026-03-27", "passenger_count": 1, "fare_class": "main_cabin"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    journey_ids = [opt["journey_id"] for opt in result["options"]]
    assert "FL_AA200_20260327" in journey_ids

    # Step 3: rebook from AA100 to AA200 (voluntary)
    result = rebook_flight(
        {
            "confirmation_number": "CONF01",
            "journey_id": "FL_AA100_20260327",
            "new_journey_id": "FL_AA200_20260327",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 75  # main_cabin voluntary fee
    assert result["cost_summary"]["fare_difference"] == 50  # 400 - 350
    assert result["cost_summary"]["total_collected"] == 125  # 75 + 50

    # Step 4: verify db mutations
    bookings = db["reservations"]["CONF01"]["bookings"]
    aa100 = next(b for b in bookings if b["journey_id"] == "FL_AA100_20260327")
    aa200 = next(b for b in bookings if b["journey_id"] == "FL_AA200_20260327")
    assert aa100["status"] == "cancelled"
    assert aa200["status"] == "confirmed"

    # Step 5: assign window seat on new booking
    result = assign_seat(
        {
            "confirmation_number": "CONF01",
            "passenger_id": "PAX001",
            "journey_id": "FL_AA200_20260327",
            "seat_preference": "window",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["seat_assigned"] == "21A"  # base_row=20 + passenger_index=1, window→A

    # Step 6: verify seat mutation on confirmed booking
    aa200_refreshed = next(
        b for b in db["reservations"]["CONF01"]["bookings"] if b["journey_id"] == "FL_AA200_20260327"
    )
    assert aa200_refreshed["segments"][0]["seat"] == "21A"

    # Step 7: add 2 checked bags
    result = add_baggage_allowance(
        {"confirmation_number": "CONF01", "journey_id": "FL_AA200_20260327", "num_bags": 2},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["bags_checked"] == 2

    # Step 8: verify baggage mutation
    aa200_refreshed = next(
        b for b in db["reservations"]["CONF01"]["bookings"] if b["journey_id"] == "FL_AA200_20260327"
    )
    assert aa200_refreshed["segments"][0]["bags_checked"] == 2


def test_flow_irrops_cancel_and_refund(fresh_db):
    """CONF02: lookup → flight status → disruption → meal voucher → cancel → refund."""
    db = fresh_db

    # Step 1: get reservation
    result = get_reservation({"confirmation_number": "CONF02", "last_name": "Doe"}, db, call_index=1)
    assert result["status"] == "success"

    # Step 2: check flight status
    result = get_flight_status({"flight_number": "BB100", "flight_date": "2026-03-29"}, db, call_index=1)
    assert result["status"] == "success"
    assert result["journey"]["status"] == "cancelled"

    # Step 3: get disruption info
    result = get_disruption_info({"flight_number": "BB100", "date": "2026-03-29"}, db, call_index=1)
    assert result["status"] == "success"
    assert result["disruption"]["is_irrops"] is True
    assert result["disruption"]["passenger_entitled_to"]["refund_option"] is True

    # Step 4: issue meal voucher
    result = issue_meal_voucher(
        {"confirmation_number": "CONF02", "passenger_id": "PAX001", "voucher_reason": "cancellation_wait_same_day"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["amount"] == 15
    assert "MEAL-CONF02" in result["voucher_code"]
    assert result["voucher_code"] in db["meal_vouchers"]

    # Step 5: cancel with irrops refund
    result = cancel_reservation(
        {"confirmation_number": "CONF02", "journey_id": "FL_BB100_20260329", "cancellation_reason": "irrops_refund"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["is_refundable"] is True
    assert result["refund_amount_eligible"] == 280.0
    assert result["credit_amount_eligible"] == 0

    # Step 6: verify db mutations
    bb100 = next(b for b in db["reservations"]["CONF02"]["bookings"] if b["journey_id"] == "FL_BB100_20260329")
    assert bb100["status"] == "cancelled"
    assert db["reservations"]["CONF02"]["status"] == "cancelled"  # only booking cancelled → reservation cancelled

    # Step 7: process refund
    result = process_refund(
        {"confirmation_number": "CONF02", "refund_amount": 280.0, "refund_type": "full_fare"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["refund_id"] == "REF-CONF02-001"
    assert result["processing_days"] == 7

    # Step 8: verify refund stored in db
    assert "REF-CONF02-001" in db["refunds"]
    assert db["refunds"]["REF-CONF02-001"]["refund_amount"] == 280.0
    assert db["refunds"]["REF-CONF02-001"]["initiated_date"] == "2026-02-19"


def test_flow_irrops_rebook_no_fees(fresh_db):
    """CONF02: disruption → search → irrops rebook (no fees) → assign seat → add baggage."""
    db = fresh_db

    # Step 1: get disruption info
    result = get_disruption_info({"flight_number": "BB100", "date": "2026-03-29"}, db, call_index=1)
    assert result["status"] == "success"
    assert result["disruption"]["is_irrops"] is True

    # Step 2: search for alternatives
    result = search_rebooking_options(
        {"origin": "SFO", "destination": "ORD", "date": "2026-03-29", "passenger_count": 1, "fare_class": "main_cabin"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert any(opt["journey_id"] == "FL_BB200_20260329" for opt in result["options"])

    # Step 3: rebook under IRROPS (no fees, no fare collection)
    result = rebook_flight(
        {
            "confirmation_number": "CONF02",
            "journey_id": "FL_BB100_20260329",
            "new_journey_id": "FL_BB200_20260329",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["cost_summary"]["change_fee"] == 0
    assert result["cost_summary"]["total_collected"] == 0
    assert result["cost_summary"]["fee_waived"] is True

    # Verify db: old cancelled, new confirmed
    bookings = db["reservations"]["CONF02"]["bookings"]
    bb100 = next(b for b in bookings if b["journey_id"] == "FL_BB100_20260329")
    bb200 = next(b for b in bookings if b["journey_id"] == "FL_BB200_20260329")
    assert bb100["status"] == "cancelled"
    assert bb200["status"] == "confirmed"

    # Step 4: assign aisle seat on new booking
    result = assign_seat(
        {
            "confirmation_number": "CONF02",
            "passenger_id": "PAX001",
            "journey_id": "FL_BB200_20260329",
            "seat_preference": "aisle",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["seat_assigned"] == "21C"  # main_cabin base_row=20 + index=1, aisle→C

    # Step 5: add baggage
    result = add_baggage_allowance(
        {"confirmation_number": "CONF02", "journey_id": "FL_BB200_20260329", "num_bags": 1},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["bags_checked"] == 1


def test_flow_partial_rebook_connecting_irrops(fresh_db):
    """CONF03: lookup → disruption → meal voucher → partial irrops rebook → assign seat on new leg."""
    db = fresh_db

    # Step 1: get reservation — should have 1 booking with 2 segments (CC300, CC400)
    result = get_reservation({"confirmation_number": "CONF03", "last_name": "Johnson"}, db, call_index=1)
    assert result["status"] == "success"
    assert len(result["reservation"]["bookings"]) == 1
    assert len(result["reservation"]["bookings"][0]["segments"]) == 2

    # Step 2: get disruption info
    result = get_disruption_info({"flight_number": "CC300", "date": "2026-03-25"}, db, call_index=1)
    assert result["status"] == "success"
    assert result["disruption"]["is_irrops"] is True
    assert result["disruption"]["passenger_entitled_to"]["hotel_accommodation"] is True

    # Step 3: issue meal voucher (stored in db)
    result = issue_meal_voucher(
        {"confirmation_number": "CONF03", "passenger_id": "PAX001", "voucher_reason": "cancellation_wait_same_day"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["amount"] == 15

    # Step 4: partial rebook — replace only CC300 leg under IRROPS (no fees)
    result = rebook_flight(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC300_CC400_20260325",
            "new_journey_id": "FL_CC350_20260325",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
            "flight_number": "CC300",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["partial_rebook"] is True
    assert result["replaced_segment"]["flight_number"] == "CC300"
    assert result["replaced_segment"]["fare_paid"] == 200.0
    assert len(result["kept_segments"]) == 1
    assert result["kept_segments"][0]["flight_number"] == "CC400"
    assert result["cost_summary"]["change_fee"] == 0
    assert result["cost_summary"]["total_collected"] == 0
    assert result["cost_summary"]["fee_waived"] is True
    assert result["cost_summary"]["fare_difference"] == 20.0  # 220 - 200 (informational only)

    # Step 5: verify db mutations
    bookings = db["reservations"]["CONF03"]["bookings"]
    original = next(b for b in bookings if b["journey_id"] == "FL_CC300_CC400_20260325" and b["status"] == "cancelled")
    cc350 = next(b for b in bookings if b["journey_id"] == "FL_CC350_20260325" and b["status"] == "confirmed")
    kept = next(b for b in bookings if b["journey_id"] == "FL_CC300_CC400_20260325" and b["status"] == "confirmed")
    assert original is not None
    assert cc350 is not None
    assert kept is not None
    assert len(kept["segments"]) == 1
    assert kept["segments"][0]["flight_number"] == "CC400"
    assert kept["segments"][0]["seat"] == "12C"  # ancillaries preserved

    # Step 6: assign window seat on new CC350 booking
    result = assign_seat(
        {
            "confirmation_number": "CONF03",
            "passenger_id": "PAX001",
            "journey_id": "FL_CC350_20260325",
            "seat_preference": "window",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["seat_assigned"] == "21A"

    # Step 7: verify seat mutation landed on the CC350 booking
    cc350_refreshed = next(
        b for b in db["reservations"]["CONF03"]["bookings"] if b["journey_id"] == "FL_CC350_20260325"
    )
    assert cc350_refreshed["segments"][0]["seat"] == "21A"


def test_flow_nonrefundable_cancel_one_leg_travel_credit(fresh_db):
    """CONF04: cancel outbound leg of round-trip → travel credit issued; return leg stays active."""
    db = fresh_db

    # Step 1: get reservation — 2 confirmed bookings
    result = get_reservation({"confirmation_number": "CONF04", "last_name": "Williams"}, db, call_index=1)
    assert result["status"] == "success"
    assert len(result["reservation"]["bookings"]) == 2

    # Step 2: cancel outbound leg (non-refundable, voluntary)
    result = cancel_reservation(
        {"confirmation_number": "CONF04", "journey_id": "FL_DD100_20260401", "cancellation_reason": "voluntary"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["is_refundable"] is False
    assert result["cancellation_fee"] == 100
    assert result["refund_amount_eligible"] == 0
    assert result["credit_amount_eligible"] == 250.0  # 350 - 100
    assert result["reservation_status"] != "cancelled"  # return leg still active

    # Step 3: verify db — DD100 cancelled, DD200 still confirmed
    dd100 = next(b for b in db["reservations"]["CONF04"]["bookings"] if b["journey_id"] == "FL_DD100_20260401")
    dd200 = next(b for b in db["reservations"]["CONF04"]["bookings"] if b["journey_id"] == "FL_DD200_20260415")
    assert dd100["status"] == "cancelled"
    assert dd200["status"] == "confirmed"

    # Step 4: issue travel credit for the forfeited fare
    result = issue_travel_credit(
        {
            "confirmation_number": "CONF04",
            "passenger_id": "PAX001",
            "amount": 250.0,
            "credit_reason": "cancellation_non_refundable",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["credit_code"] == "TCCONF04PAX"
    assert result["amount"] == 250.0

    # Step 5: verify credit stored in db
    assert "TCCONF04PAX" in db["travel_credits"]
    assert db["travel_credits"]["TCCONF04PAX"]["issued_date"] == "2026-02-19"
    assert db["travel_credits"]["TCCONF04PAX"]["expiry_date"] == "2027-02-19"


def test_flow_cancel_both_legs_reservation_fully_cancelled(fresh_db):
    """CONF04: cancel both legs sequentially → reservation reaches 'cancelled' status."""
    db = fresh_db

    # Step 1: cancel outbound leg
    result1 = cancel_reservation(
        {"confirmation_number": "CONF04", "journey_id": "FL_DD100_20260401", "cancellation_reason": "voluntary"},
        db,
        call_index=1,
    )
    assert result1["status"] == "success"
    assert result1["credit_amount_eligible"] == 250.0
    assert result1["reservation_status"] != "cancelled"  # return leg still active

    # Step 2: cancel return leg
    result2 = cancel_reservation(
        {"confirmation_number": "CONF04", "journey_id": "FL_DD200_20260415", "cancellation_reason": "voluntary"},
        db,
        call_index=1,
    )
    assert result2["status"] == "success"
    assert result2["credit_amount_eligible"] == 250.0

    # Step 3: verify db — reservation fully cancelled
    assert db["reservations"]["CONF04"]["bookings"][0]["status"] == "cancelled"
    assert db["reservations"]["CONF04"]["bookings"][1]["status"] == "cancelled"
    assert db["reservations"]["CONF04"]["status"] == "cancelled"

    # Step 4 & 5: process two separate refunds; each must get a distinct refund_id
    result_r1 = process_refund(
        {"confirmation_number": "CONF04", "refund_amount": 250.0, "refund_type": "partial_fare"},
        db,
        call_index=1,
    )
    result_r2 = process_refund(
        {"confirmation_number": "CONF04", "refund_amount": 250.0, "refund_type": "partial_fare"},
        db,
        call_index=2,
    )
    assert result_r1["status"] == "success"
    assert result_r2["status"] == "success"
    assert result_r1["refund_id"] == "REF-CONF04-001"
    assert result_r2["refund_id"] == "REF-CONF04-002"
    assert result_r1["refund_id"] != result_r2["refund_id"]

    # Both refunds stored in db
    assert len(db["refunds"]) == 2


def test_flow_standby_for_earlier_flight(fresh_db):
    """CONF01: lookup → search → add to standby; original confirmed booking unchanged."""
    db = fresh_db

    # Step 1: get reservation
    result = get_reservation({"confirmation_number": "CONF01", "last_name": "Smith"}, db, call_index=1)
    assert result["status"] == "success"

    # Step 2: search for alternatives (to surface FL_AA200)
    result = search_rebooking_options(
        {"origin": "LAX", "destination": "JFK", "date": "2026-03-27", "passenger_count": 1, "fare_class": "main_cabin"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert any(opt["journey_id"] == "FL_AA200_20260327" for opt in result["options"])

    # Step 3: add PAX001 to standby on FL_AA200
    result = add_to_standby(
        {"confirmation_number": "CONF01", "journey_id": "FL_AA200_20260327", "passenger_ids": ["PAX001"]},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["standby_list_position"] == 1

    # Step 4: verify standby list on reservation
    assert "standby_list" in db["reservations"]["CONF01"]
    assert len(db["reservations"]["CONF01"]["standby_list"]) == 1
    assert db["reservations"]["CONF01"]["standby_list"][0]["journey_id"] == "FL_AA200_20260327"

    # Step 5: verify standby list on journey
    assert "standby_list" in db["journeys"]["FL_AA200_20260327"]
    standby_entries = db["journeys"]["FL_AA200_20260327"]["standby_list"]
    assert any(entry["passenger_id"] == "PAX001" for entry in standby_entries)

    # Step 6: original FL_AA100 booking is still confirmed (standby doesn't cancel it)
    aa100 = next(b for b in db["reservations"]["CONF01"]["bookings"] if b["journey_id"] == "FL_AA100_20260327")
    assert aa100["status"] == "confirmed"


def test_flow_hotel_and_meal_voucher_missed_connection(fresh_db):
    """CONF03: lookup → disruption → meal voucher → hotel voucher; both stored in db."""
    db = fresh_db

    # Step 1: get reservation
    result = get_reservation({"confirmation_number": "CONF03", "last_name": "Johnson"}, db, call_index=1)
    assert result["status"] == "success"

    # Step 2: get disruption info confirming hotel accommodation entitlement
    result = get_disruption_info({"flight_number": "CC300", "date": "2026-03-25"}, db, call_index=1)
    assert result["status"] == "success"
    assert result["disruption"]["passenger_entitled_to"]["hotel_accommodation"] is True

    # Step 3: issue meal voucher for overnight missed connection ($25)
    result = issue_meal_voucher(
        {"confirmation_number": "CONF03", "passenger_id": "PAX001", "voucher_reason": "irrops_overnight"},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["amount"] == 25
    meal_code = result["voucher_code"]

    # Step 4: issue hotel voucher for 1 night
    result = issue_hotel_voucher(
        {"confirmation_number": "CONF03", "passenger_id": "PAX001", "num_nights": 1},
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["voucher_code"] == "HOTEL-CONF03"
    assert result["number_of_nights"] == 1

    # Step 5: verify both vouchers stored in db
    assert meal_code in db["meal_vouchers"]
    assert db["meal_vouchers"][meal_code]["amount"] == 25
    assert "HOTEL-CONF03" in db.get("hotel_vouchers", {})
    assert db["hotel_vouchers"]["HOTEL-CONF03"]["num_nights"] == 1
    assert db["hotel_vouchers"]["HOTEL-CONF03"]["issued_date"] == "2026-02-19"


def test_flow_multi_search_no_basic_economy_then_rebook(fresh_db):
    """CONF01: search basic_economy returns nothing, search main_cabin succeeds → rebook."""
    db = fresh_db

    # Step 1: get reservation
    result = get_reservation({"confirmation_number": "CONF01", "last_name": "Smith"}, db, call_index=1)
    assert result["status"] == "success"

    # Step 2: search basic_economy — FL_AA100 and FL_AA200 have 0 basic_economy seats
    result = search_rebooking_options(
        {
            "origin": "LAX",
            "destination": "JFK",
            "date": "2026-03-27",
            "passenger_count": 1,
            "fare_class": "basic_economy",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["count"] == 0  # no basic_economy options available

    # Step 3: search main_cabin — FL_AA100 and FL_AA200 both available
    result = search_rebooking_options(
        {"origin": "LAX", "destination": "JFK", "date": "2026-03-27", "passenger_count": 1, "fare_class": "main_cabin"},
        db,
        call_index=2,
    )
    assert result["status"] == "success"
    assert result["count"] >= 1
    assert any(opt["journey_id"] == "FL_AA200_20260327" for opt in result["options"])

    # Step 4: rebook to FL_AA200
    result = rebook_flight(
        {
            "confirmation_number": "CONF01",
            "journey_id": "FL_AA100_20260327",
            "new_journey_id": "FL_AA200_20260327",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"

    # Step 5: verify final db state after two searches + rebook
    bookings = db["reservations"]["CONF01"]["bookings"]
    assert any(b["journey_id"] == "FL_AA100_20260327" and b["status"] == "cancelled" for b in bookings)
    assert any(b["journey_id"] == "FL_AA200_20260327" and b["status"] == "confirmed" for b in bookings)


def test_bug_meal_voucher_no_reservation_validation(fresh_db):
    """Bug A: issue_meal_voucher never validates the reservation exists.

    Expected: error with error_type='not_found' for bogus confirmation number.
    Actual: returns success (no _lookup_reservation call in issue_meal_voucher).
    """
    result = issue_meal_voucher(
        {"confirmation_number": "BOGUS9", "passenger_id": "PAX001", "voucher_reason": "cancellation_wait_same_day"},
        fresh_db,
        call_index=1,
    )
    assert result["status"] == "error", (
        f"Bug A: issue_meal_voucher accepted BOGUS9 (no reservation validation). Got: {result}"
    )
    assert result["error_type"] == "not_found"


def test_bug_partial_rebook_kept_booking_mutation_target(fresh_db):
    """Verify partial rebook targets the correct booking for mutations.

    Bug B: after partial rebook, _find_booking_journey returns the CANCELLED original
    (linear scan, first match) instead of the CONFIRMED kept booking. Mutations
    from add_baggage_allowance go to the wrong (cancelled) booking.

    Expected: bags_checked=2 on the CONFIRMED kept booking's CC400 segment.
    Actual: mutation lands on the CANCELLED original's CC400 segment.
    """
    db = fresh_db

    # Step 1: partial rebook — cancel CC300, keep CC400, add CC350
    rebook_flight(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC300_CC400_20260325",
            "new_journey_id": "FL_CC350_20260325",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
            "flight_number": "CC300",
        },
        db,
        call_index=1,
    )

    # Step 2: add baggage to the kept CC400 leg (journey_id is the shared one)
    result = add_baggage_allowance(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC300_CC400_20260325",
            "flight_number": "CC400",
            "num_bags": 2,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"

    # Step 3: verify mutation landed on the CONFIRMED kept booking, not the cancelled one
    bookings = db["reservations"]["CONF03"]["bookings"]
    kept = next(b for b in bookings if b["journey_id"] == "FL_CC300_CC400_20260325" and b["status"] == "confirmed")
    cancelled = next(b for b in bookings if b["journey_id"] == "FL_CC300_CC400_20260325" and b["status"] == "cancelled")
    cc400_kept = next(s for s in kept["segments"] if s["flight_number"] == "CC400")
    cc400_cancelled = next(s for s in cancelled["segments"] if s["flight_number"] == "CC400")

    assert cc400_kept["bags_checked"] == 2, (
        f"Bug B: mutation went to cancelled booking. kept has {cc400_kept['bags_checked']}, "
        f"cancelled has {cc400_cancelled['bags_checked']}"
    )
    assert cc400_cancelled["bags_checked"] == 1, (
        "Bug B: original cancelled booking's CC400 segment was incorrectly mutated"
    )


def test_bug_kept_booking_cancel_returns_correct_credit(fresh_db):
    """Verify kept booking cancel returns correct credit after partial rebook.

    Bug B+C: after partial rebook, cancelling the kept booking fails because
    _find_booking_journey returns the already-cancelled original (first match),
    triggering an 'already_cancelled' error instead of processing the cancel.

    Expected: status='success', credit_amount_eligible=150 (CC400 fare 250 - 100 fee).
    Actual: status='error', error_type='already_cancelled' (wrong booking found).
    """
    db = fresh_db

    # Step 1: partial rebook — CC300 replaced, CC400 kept under same journey_id
    rebook_flight(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC300_CC400_20260325",
            "new_journey_id": "FL_CC350_20260325",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
            "flight_number": "CC300",
        },
        db,
        call_index=1,
    )

    # Step 2: cancel the kept booking (only CC400 segment, fare=250)
    result = cancel_reservation(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC300_CC400_20260325",
            "cancellation_reason": "voluntary",
        },
        db,
        call_index=1,
    )

    # Expected: the CONFIRMED kept booking is found and cancelled
    assert result["status"] == "success", (
        f"Bug B/C: cancel_reservation found the already-cancelled original instead of the "
        f"confirmed kept booking. Got: {result}"
    )
    # Non-refundable: credit = kept_fare(250) - cancellation_fee(100) = 150
    assert result["credit_amount_eligible"] == 150.0, (
        f"Bug C: credit computed from wrong booking. Expected 150, got {result.get('credit_amount_eligible')}"
    )


def test_bug_standby_position_second_call_same_flight(fresh_db):
    """Bug D: standby_list_position = len(passenger_ids), not the queue position.

    When 1 passenger is added, len([PAX001]) = 1, which happens to be correct on
    the first call (empty queue). On the second call to the SAME flight, the queue
    already has 1 entry so position should be 2, but the bug returns 1 again.

    Expected: second call returns standby_list_position=2.
    Actual: returns standby_list_position=1 (len of passenger_ids list).
    """
    db = fresh_db

    # First call — queue is empty, position should be 1 (coincidentally correct)
    result1 = add_to_standby(
        {"confirmation_number": "CONF01", "journey_id": "FL_AA200_20260327", "passenger_ids": ["PAX001"]},
        db,
        call_index=1,
    )
    assert result1["status"] == "success"
    assert result1["standby_list_position"] == 1  # correct by coincidence

    # Second call to the SAME flight — queue now has 1 entry, position should be 2
    result2 = add_to_standby(
        {"confirmation_number": "CONF01", "journey_id": "FL_AA200_20260327", "passenger_ids": ["PAX001"]},
        db,
        call_index=2,
    )
    assert result2["status"] == "success"
    assert result2["standby_list_position"] == 2, (
        f"Bug D: standby_list_position = len(passenger_ids) = 1, not queue position 2. "
        f"Got: {result2['standby_list_position']}"
    )


def test_bug_irrops_rebook_new_booking_fare_paid_should_be_old_fare(fresh_db):
    """Bug E: IRROPS rebook sets new_booking.fare_paid = new flight fare instead of old_fare.

    Under IRROPS, the passenger's original fare investment carries over to the new booking.
    fare_paid should equal old_fare (280.0 for CONF02), not the market fare of the replacement flight.

    Expected: new_booking.fare_paid = 280.0 (old_fare)
    Actual (before fix): new_booking.fare_paid = 300 (full BB200 main_cabin fare)

    Also verifies that cancelling the new booking with irrops_refund returns the correct
    refund amount (280.0 = original investment, not the BB200 market fare of 300.0).
    """
    db = fresh_db

    result = rebook_flight(
        {
            "confirmation_number": "CONF02",
            "journey_id": "FL_BB100_20260329",
            "new_journey_id": "FL_BB200_20260329",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["cost_summary"]["total_collected"] == 0  # sanity: IRROPS = no charge

    # The new booking's fare_paid should equal old_fare (original investment carries over)
    bookings = db["reservations"]["CONF02"]["bookings"]
    new_booking = next(b for b in bookings if b["journey_id"] == "FL_BB200_20260329" and b["status"] == "confirmed")
    assert new_booking["fare_paid"] == 280.0, (
        f"Bug E: IRROPS rebook set fare_paid = {new_booking['fare_paid']} instead of old_fare 280.0"
    )

    # Cancelling the rebooked flight should refund the original investment, not the market fare
    result = cancel_reservation(
        {
            "confirmation_number": "CONF02",
            "journey_id": "FL_BB200_20260329",
            "cancellation_reason": "irrops_refund",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["is_refundable"] is True
    assert result["refund_amount_eligible"] == 280.0, (
        f"Bug E: irrops_refund returned {result['refund_amount_eligible']} instead of "
        f"original investment 280.0 (would be 300.0 if fare_paid was wrongly set to BB200 market fare)"
    )
    assert result["credit_amount_eligible"] == 0


def test_bug_voluntary_rebook_new_booking_fare_paid_should_be_new_fare(fresh_db):
    """Bug F: Voluntary rebook sets new_booking.fare_paid = new_fare (total investment).

    CONF01: AA100 main_cabin=$350, AA200 main_cabin=$400 → new_fare=$400.
    The new booking fare_paid should equal new_fare ($400), the passenger's total
    investment in the replacement ticket.

    Expected: new_booking.fare_paid = 400
    Actual (before fix): new_booking.fare_paid = 400 (already correct for voluntary)

    Also verifies that cancelling the new booking returns a travel credit based on the
    correct fare_paid (400 - 100 cancellation fee = 300 credit).
    """
    db = fresh_db

    result = rebook_flight(
        {
            "confirmation_number": "CONF01",
            "journey_id": "FL_AA100_20260327",
            "new_journey_id": "FL_AA200_20260327",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["cost_summary"]["fare_difference"] == 50  # sanity check

    bookings = db["reservations"]["CONF01"]["bookings"]
    new_booking = next(b for b in bookings if b["journey_id"] == "FL_AA200_20260327" and b["status"] == "confirmed")
    assert new_booking["fare_paid"] == 400.0, (
        f"Bug F: voluntary rebook set fare_paid = {new_booking['fare_paid']} instead of new_fare 400.0"
    )

    # Cancelling the rebooked flight should issue credit based on the full new fare paid
    result = cancel_reservation(
        {
            "confirmation_number": "CONF01",
            "journey_id": "FL_AA200_20260327",
            "cancellation_reason": "voluntary",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["is_refundable"] is False
    assert result["cancellation_fee"] == 100
    assert result["refund_amount_eligible"] == 0
    assert result["credit_amount_eligible"] == 300.0, (
        f"Bug F: cancel returned credit_amount_eligible={result['credit_amount_eligible']} instead of 300.0 "
        f"(fare_paid 400.0 - cancellation_fee 100)"
    )


def test_bug_voluntary_rebook_cheaper_flight_fare_paid_should_be_new_fare(fresh_db):
    """Bug F (variant): Voluntary rebook to a cheaper flight sets fare_paid = new_fare.

    For voluntary rebooks, fare_paid = new_fare regardless of direction.
    CONF01: rebook AA200($400) → AA100($350) means new_fare = $350.
    New booking fare_paid should be 350.0.

    Also verifies that cancelling the new booking returns a travel credit based on the
    correct fare_paid (350 - 100 cancellation fee = 250 credit).
    """
    db = fresh_db

    # First rebook to AA200 so we can then rebook back to AA100 (cheaper)
    rebook_flight(
        {
            "confirmation_number": "CONF01",
            "journey_id": "FL_AA100_20260327",
            "new_journey_id": "FL_AA200_20260327",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )

    # Now rebook from AA200 back to AA100 (cheaper: $400 → $350, diff = -50)
    result = rebook_flight(
        {
            "confirmation_number": "CONF01",
            "journey_id": "FL_AA200_20260327",
            "new_journey_id": "FL_AA100_20260327",
            "rebooking_type": "voluntary",
            "waive_change_fee": False,
        },
        db,
        call_index=2,
    )
    assert result["status"] == "success"
    assert result["cost_summary"]["fare_difference"] < 0  # cheaper flight

    bookings = db["reservations"]["CONF01"]["bookings"]
    new_booking = next(b for b in bookings if b["journey_id"] == "FL_AA100_20260327" and b["status"] == "confirmed")
    assert new_booking["fare_paid"] == 350.0, (
        f"Bug F: rebook to cheaper flight set fare_paid = {new_booking['fare_paid']} instead of new_fare 350.0"
    )

    # Cancelling should credit based on the actual fare paid (350), not a stale value
    result = cancel_reservation(
        {
            "confirmation_number": "CONF01",
            "journey_id": "FL_AA100_20260327",
            "cancellation_reason": "voluntary",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"
    assert result["is_refundable"] is False
    assert result["cancellation_fee"] == 100
    assert result["refund_amount_eligible"] == 0
    assert result["credit_amount_eligible"] == 250.0, (
        f"Bug F: cancel returned credit_amount_eligible={result['credit_amount_eligible']} instead of 250.0 "
        f"(fare_paid 350.0 - cancellation_fee 100)"
    )


def test_irrops_partial_rebook_preserves_segment_fare_paid(fresh_db):
    """Verify IRROPS partial rebook preserves old segment fare_paid through multiple rebooks.

    When rebook_flight creates a new IRROPS booking, segment-level fare_paid now matches
    the old segment's fare_paid (not the new flight's market fare). This ensures that
    subsequent rebooks and cancellations use the correct passenger investment.

    Scenario:
      1. IRROPS partial rebook: CC300 (orig fare=200) -> CC350 (market fare=220)
         new_booking FL_CC350: journey.fare_paid=200, segment.fare_paid=200 (preserved)
      2. Second IRROPS rebook: CC350 -> CC360 (market fare=230), flight_number="CC350"
         old_fare used: 200 (segment-level, correct)
         CC360 new_booking.fare_paid = 200 (correct)
      3. Cancel CC360 with irrops_refund:
         refund_amount_eligible = 200 (correct, passenger's original investment)
    """
    db = fresh_db

    # Add a second LAX->ORD alternative so CC350 can be rebooked to it
    db["journeys"]["FL_CC360_20260325"] = {
        "journey_id": "FL_CC360_20260325",
        "date": "2026-03-25",
        "origin": "LAX",
        "destination": "ORD",
        "num_stops": 0,
        "total_duration_minutes": 240,
        "fares": {
            "basic_economy": None,
            "main_cabin": 230.0,
            "premium_economy": 380.0,
            "business": None,
            "first": None,
        },
        "segments": [
            {
                "segment_number": 1,
                "flight_number": "CC360",
                "origin": "LAX",
                "destination": "ORD",
                "scheduled_departure": "11:00",
                "origin_utc_offset": -8,
                "scheduled_arrival": "17:00",
                "destination_utc_offset": -6,
                "duration_minutes": 240,
                "aircraft_type": "737-800",
                "status": "scheduled",
                "delay_minutes": None,
                "delay_reason": None,
                "cancellation_reason": None,
                "gate": "D9",
                "available_seats": {
                    "basic_economy": 0,
                    "main_cabin": 15,
                    "premium_economy": 5,
                    "business": 0,
                    "first": 0,
                },
                "fares": {
                    "basic_economy": None,
                    "main_cabin": 230.0,
                    "premium_economy": 380.0,
                    "business": None,
                    "first": None,
                },
            }
        ],
        "status": "scheduled",
        "bookable": True,
    }

    # Step 1: IRROPS partial rebook -- replace CC300 (orig fare=200) with CC350 (market=220)
    result = rebook_flight(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC300_CC400_20260325",
            "new_journey_id": "FL_CC350_20260325",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
            "flight_number": "CC300",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success", f"Step 1 setup failed: {result}"

    # Confirm segment-level fare_paid is preserved (not stale market fare)
    bookings = db["reservations"]["CONF03"]["bookings"]
    cc350_booking = next(b for b in bookings if b["journey_id"] == "FL_CC350_20260325")
    assert cc350_booking["fare_paid"] == 200.0, (
        f"journey-level fare_paid should be 200 (IRROPS old_fare), got {cc350_booking['fare_paid']}"
    )
    assert cc350_booking["segments"][0]["fare_paid"] == 200.0, (
        f"segment-level fare_paid should be 200 (preserved from old segment), "
        f"got {cc350_booking['segments'][0]['fare_paid']}"
    )

    # Step 2: second IRROPS rebook CC350 -> CC360, passing flight_number="CC350"
    # segment-level old_fare = 200 (correct, preserved from original investment)
    result = rebook_flight(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC350_20260325",
            "new_journey_id": "FL_CC360_20260325",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
            "flight_number": "CC350",
        },
        db,
        call_index=2,
    )
    assert result["status"] == "success", f"Step 2 rebook failed: {result}"

    # Step 3: passenger cancels CC360 with irrops_refund
    # cancel_reservation uses _get_booking_total_fare -> journey-level fare_paid = 200 (correct)
    result = cancel_reservation(
        {
            "confirmation_number": "CONF03",
            "journey_id": "FL_CC360_20260325",
            "cancellation_reason": "irrops_refund",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success", f"Step 3 cancel failed: {result}"
    assert result["is_refundable"] is True

    assert result["refund_amount_eligible"] == 200.0, (
        f"refund_amount_eligible should be 200 (passenger's original IRROPS investment), but got {result['refund_amount_eligible']}"
    )


def test_cancel_reservation_increments_seats(fresh_db):
    """Cancelling a booking should increment available_seats on the cancelled journey's segments."""
    db = fresh_db

    # Check initial seat count for BB200 main_cabin
    bb200_seg = db["journeys"]["FL_BB200_20260329"]["segments"][0]
    initial_seats = bb200_seg["available_seats"]["main_cabin"]

    # First rebook CONF02 to BB200 so we have a confirmed booking on it
    result = rebook_flight(
        {
            "confirmation_number": "CONF02",
            "journey_id": "FL_BB100_20260329",
            "new_journey_id": "FL_BB200_20260329",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"

    # Rebook decrements seats by 1
    assert bb200_seg["available_seats"]["main_cabin"] == initial_seats - 1

    # Now cancel the BB200 booking
    result = cancel_reservation(
        {
            "confirmation_number": "CONF02",
            "journey_id": "FL_BB200_20260329",
            "cancellation_reason": "voluntary",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"

    # Cancel should increment seats back by 1
    assert bb200_seg["available_seats"]["main_cabin"] == initial_seats


def test_cancel_reservation_seats_0_to_1(sample_db):
    """Cancelling a booking on a flight with 0 available seats should increment to 1."""
    db = copy.deepcopy(sample_db)

    # SW100 has 0 seats in main_cabin (it's cancelled)
    # But we have a booking on it to cancel — set it to confirmed for this test
    sw100_seg = db["journeys"]["FL_SW100_20260320"]["segments"][0]
    assert sw100_seg["available_seats"]["main_cabin"] == 0

    # Make the SW100 booking confirmed so we can cancel it
    booking = db["reservations"]["ABC123"]["bookings"][0]
    assert booking["journey_id"] == "FL_SW100_20260320"
    booking["status"] = "confirmed"

    result = cancel_reservation(
        {
            "confirmation_number": "ABC123",
            "journey_id": "FL_SW100_20260320",
            "cancellation_reason": "voluntary",
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"

    # Should go from 0 to 1
    assert sw100_seg["available_seats"]["main_cabin"] == 1


def test_irrops_rebook_segment_fare_paid_matches_old(fresh_db):
    """IRROPS full rebook should set segment fare_paid = old segment fare_paid, not market fare."""
    db = fresh_db

    # CONF02: BB100 booking has segment fare_paid=280.0
    old_booking = next(b for b in db["reservations"]["CONF02"]["bookings"] if b["journey_id"] == "FL_BB100_20260329")
    old_segment_fare = old_booking["segments"][0]["fare_paid"]
    assert old_segment_fare == 280.0

    # BB200 market fare in main_cabin is 300.0
    bb200_market_fare = db["journeys"]["FL_BB200_20260329"]["segments"][0]["fares"]["main_cabin"]
    assert bb200_market_fare == 300.0

    # IRROPS full rebook
    result = rebook_flight(
        {
            "confirmation_number": "CONF02",
            "journey_id": "FL_BB100_20260329",
            "new_journey_id": "FL_BB200_20260329",
            "rebooking_type": "irrops_cancellation",
            "waive_change_fee": False,
        },
        db,
        call_index=1,
    )
    assert result["status"] == "success"

    # New booking segment fare_paid should be old segment fare (280.0), not market fare (300.0)
    new_booking = next(b for b in db["reservations"]["CONF02"]["bookings"] if b["journey_id"] == "FL_BB200_20260329")
    assert new_booking["fare_paid"] == 280.0, "booking-level fare_paid should be old_fare"
    assert new_booking["segments"][0]["fare_paid"] == 280.0, (
        f"segment fare_paid should be old segment fare (280.0), got {new_booking['segments'][0]['fare_paid']}"
    )


class TestConfirmationNumberFormat:
    """Validate confirmation_number format across tools."""

    @pytest.mark.parametrize(
        "bad_value,reason",
        [
            ("ABC", "too short"),
            ("ABC12345", "too long"),
            ("ABC-12", "contains dash"),
            ("ABC 12", "contains space"),
            ("ABC!23", "contains special character"),
        ],
    )
    def test_get_reservation_bad_confirmation_number(self, sample_db, bad_value, reason):
        result = get_reservation({"confirmation_number": bad_value, "last_name": "Doe"}, sample_db, call_index=1)
        assert result["status"] == "error", f"Should reject confirmation_number that is {reason}"
        assert result["error_type"] == "invalid_confirmation_number_format"
        assert "6 alphanumeric" in result["message"]

    @pytest.mark.parametrize(
        "bad_value",
        ["SHORT", "TOOLONGVALUE", "AB-123"],
    )
    def test_rebook_flight_bad_confirmation_number(self, sample_db, bad_value):
        result = rebook_flight(
            {
                "confirmation_number": bad_value,
                "journey_id": "FL_SW200_20260320",
                "new_journey_id": "FL_SW100_20260320",
                "rebooking_type": "voluntary",
                "waive_change_fee": False,
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_confirmation_number_format"

    def test_cancel_reservation_bad_confirmation_number(self, sample_db):
        result = cancel_reservation(
            {"confirmation_number": "X", "journey_id": "FL_SW200_20260320", "cancellation_reason": "voluntary"},
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_confirmation_number_format"


class TestFlightNumberFormat:
    """Validate flight_number format."""

    @pytest.mark.parametrize(
        "bad_value,reason",
        [
            ("123", "digits only"),
            ("ABCDE", "letters only"),
            ("S1", "single letter prefix"),
            ("SW", "no digits"),
            ("SW12345", "too many digits"),
        ],
    )
    def test_get_flight_status_bad_flight_number(self, sample_db, bad_value, reason):
        result = get_flight_status({"flight_number": bad_value, "flight_date": "2026-03-20"}, sample_db, call_index=1)
        assert result["status"] == "error", f"Should reject flight_number that has {reason}"
        assert result["error_type"] == "invalid_flight_number_format"

    def test_get_disruption_info_bad_flight_number(self, sample_db):
        result = get_disruption_info({"flight_number": "12345", "date": "2026-03-20"}, sample_db, call_index=1)
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_flight_number_format"

    def test_valid_flight_numbers_accepted(self, sample_db):
        """Ensure valid formats pass validation (may fail on lookup, but not format)."""
        for fn in ["SK1", "SW12", "ABC1234"]:
            result = get_flight_status({"flight_number": fn, "flight_date": "2026-03-20"}, sample_db, call_index=1)
            assert result.get("error_type") != "invalid_flight_number_format", (
                f"'{fn}' should be a valid flight number format"
            )


class TestDateFormat:
    """Validate date format (YYYY-MM-DD)."""

    @pytest.mark.parametrize(
        "bad_value,reason",
        [
            ("03-20-2026", "US format"),
            ("20260320", "no dashes"),
            ("2026/03/20", "slashes"),
            ("March 20, 2026", "human readable"),
            ("20-03-2026", "DD-MM-YYYY"),
        ],
    )
    def test_get_flight_status_bad_date(self, sample_db, bad_value, reason):
        result = get_flight_status({"flight_number": "SW100", "flight_date": bad_value}, sample_db, call_index=1)
        assert result["status"] == "error", f"Should reject date in {reason}"
        assert result["error_type"] == "invalid_date_format"

    def test_search_rebooking_options_bad_date(self, sample_db):
        result = search_rebooking_options(
            {
                "origin": "JFK",
                "destination": "LAX",
                "date": "March 2026",
                "passenger_count": 1,
                "fare_class": "any",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_date_format"


class TestJourneyIdFormat:
    """Validate journey_id format (FL_<flight(s)>_YYYYMMDD)."""

    @pytest.mark.parametrize(
        "bad_value,reason",
        [
            ("SW200_20260320", "missing FL_ prefix"),
            ("FL_20260320", "missing flight number"),
            ("FL_SW200", "missing date"),
            ("FL-SW200-20260320", "dashes instead of underscores"),
            ("journey_123", "completely wrong format"),
        ],
    )
    def test_rebook_flight_bad_journey_id(self, sample_db, bad_value, reason):
        result = rebook_flight(
            {
                "confirmation_number": "ABC123",
                "journey_id": bad_value,
                "new_journey_id": "FL_SW200_20260320",
                "rebooking_type": "voluntary",
                "waive_change_fee": False,
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error", f"Should reject journey_id that is {reason}"
        assert result["error_type"] == "invalid_journey_id_format"

    def test_multi_segment_journey_id_accepted(self, sample_db):
        """Multi-segment journey IDs like FL_SW300_SW400_20260322 should pass format validation."""
        result = assign_seat(
            {
                "confirmation_number": "ABC123",
                "passenger_id": "PAX001",
                "journey_id": "FL_SW300_SW400_20260322",
                "seat_preference": "window",
            },
            sample_db,
            call_index=1,
        )
        assert result.get("error_type") != "invalid_journey_id_format"


class TestPassengerIdFormat:
    """Validate passenger_id format (PAX<digits>)."""

    @pytest.mark.parametrize(
        "bad_value,reason",
        [
            ("001", "missing PAX prefix"),
            ("PASSENGER1", "wrong prefix"),
            ("PAX", "no digits"),
            ("pax001", "lowercase"),
        ],
    )
    def test_assign_seat_bad_passenger_id(self, sample_db, bad_value, reason):
        result = assign_seat(
            {
                "confirmation_number": "ABC123",
                "passenger_id": bad_value,
                "journey_id": "FL_SW200_20260320",
                "seat_preference": "window",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error", f"Should reject passenger_id that is {reason}"
        assert result["error_type"] == "invalid_passenger_id_format"

    def test_issue_travel_credit_bad_passenger_id(self, sample_db):
        result = issue_travel_credit(
            {
                "confirmation_number": "ABC123",
                "passenger_id": "passenger_1",
                "amount": 100.0,
                "credit_reason": "goodwill",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_passenger_id_format"

    def test_add_to_standby_bad_passenger_ids(self, sample_db):
        result = add_to_standby(
            {
                "confirmation_number": "ABC123",
                "journey_id": "FL_SW200_20260320",
                "passenger_ids": ["INVALID"],
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_passenger_id_format"


class TestAirportCodeFormat:
    """Validate airport code format (3 letters)."""

    @pytest.mark.parametrize(
        "bad_origin,reason",
        [
            ("JF", "too short"),
            ("JFKK", "too long"),
            ("123", "digits"),
            ("J F", "contains space"),
        ],
    )
    def test_search_rebooking_bad_origin(self, sample_db, bad_origin, reason):
        result = search_rebooking_options(
            {
                "origin": bad_origin,
                "destination": "LAX",
                "date": "2026-03-20",
                "passenger_count": 1,
                "fare_class": "any",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error", f"Should reject origin that is {reason}"
        assert result["error_type"] == "invalid_airport_code_format"

    def test_search_rebooking_bad_destination(self, sample_db):
        result = search_rebooking_options(
            {
                "origin": "JFK",
                "destination": "LA",
                "date": "2026-03-20",
                "passenger_count": 1,
                "fare_class": "any",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_airport_code_format"


class TestNumericConstraints:
    """Validate numeric field constraints."""

    def test_search_rebooking_zero_passengers(self, sample_db):
        result = search_rebooking_options(
            {
                "origin": "JFK",
                "destination": "LAX",
                "date": "2026-03-20",
                "passenger_count": 0,
                "fare_class": "any",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"

    def test_search_rebooking_negative_passengers(self, sample_db):
        result = search_rebooking_options(
            {
                "origin": "JFK",
                "destination": "LAX",
                "date": "2026-03-20",
                "passenger_count": -1,
                "fare_class": "any",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"

    def test_add_baggage_negative_count(self, sample_db):
        result = add_baggage_allowance(
            {"confirmation_number": "ABC123", "journey_id": "FL_SW200_20260320", "num_bags": -1},
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"
        assert result["error_type"] == "invalid_bag_count"

    def test_issue_hotel_voucher_zero_nights(self, sample_db):
        result = issue_hotel_voucher(
            {"confirmation_number": "ABC123", "passenger_id": "PAX001", "num_nights": 0},
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"

    def test_issue_travel_credit_zero_amount(self, sample_db):
        result = issue_travel_credit(
            {
                "confirmation_number": "ABC123",
                "passenger_id": "PAX001",
                "amount": 0,
                "credit_reason": "goodwill",
            },
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"

    def test_process_refund_negative_amount(self, sample_db):
        result = process_refund(
            {"confirmation_number": "ABC123", "refund_amount": -50.0, "refund_type": "full_fare"},
            sample_db,
            call_index=1,
        )
        assert result["status"] == "error"


class TestErrorMessageDetail:
    """Verify that validation error messages include helpful detail for the LLM."""

    def test_confirmation_number_message_includes_format_hint(self, sample_db):
        result = get_reservation({"confirmation_number": "ABC", "last_name": "Doe"}, sample_db, call_index=1)
        assert "6 alphanumeric" in result["message"]

    def test_flight_number_message_includes_format_hint(self, sample_db):
        result = get_flight_status({"flight_number": "12345", "flight_date": "2026-03-20"}, sample_db, call_index=1)
        assert "2-3 letters" in result["message"]

    def test_date_message_includes_format_hint(self, sample_db):
        result = get_flight_status({"flight_number": "SW100", "flight_date": "03-20-2026"}, sample_db, call_index=1)
        assert "YYYY-MM-DD" in result["message"]

    def test_journey_id_message_includes_format_hint(self, sample_db):
        result = cancel_reservation(
            {"confirmation_number": "ABC123", "journey_id": "bad_id", "cancellation_reason": "voluntary"},
            sample_db,
            call_index=1,
        )
        assert "FL_" in result["message"]

    def test_passenger_id_message_includes_format_hint(self, sample_db):
        result = assign_seat(
            {
                "confirmation_number": "ABC123",
                "passenger_id": "invalid",
                "journey_id": "FL_SW200_20260320",
                "seat_preference": "window",
            },
            sample_db,
            call_index=1,
        )
        assert "PAX" in result["message"]

    def test_airport_code_message_includes_format_hint(self, sample_db):
        result = search_rebooking_options(
            {
                "origin": "XX",
                "destination": "LAX",
                "date": "2026-03-20",
                "passenger_count": 1,
                "fare_class": "any",
            },
            sample_db,
            call_index=1,
        )
        assert "3-letter" in result["message"]
