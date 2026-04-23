"""Flight rebooking agent tool functions.

Each tool is implemented as a Python function that:
- Takes params, db, and call_index as arguments
- Queries/mutates the scenario database directly
- Returns a response dict with status field

Tool functions replace YAML mock blocks with explicit Python logic.
"""

import copy

from pydantic import ValidationError

from eva.assistant.tools.airline_params import (
    AddBaggageAllowanceParams,
    AddMealRequestParams,
    AddToStandbyParams,
    AssignSeatParams,
    CancelReservationParams,
    GetDisruptionInfoParams,
    GetFlightStatusParams,
    GetReservationParams,
    IssueHotelVoucherParams,
    IssueMealVoucherParams,
    IssueTravelCreditParams,
    ProcessRefundParams,
    RebookFlightParams,
    SearchRebookingOptionsParams,
    TransferToAgentParams,
    validation_error_response,
)


def _lookup_reservation(db: dict, confirmation_number: str) -> dict | None:
    """Find a reservation by confirmation number."""
    return db.get("reservations", {}).get(confirmation_number.upper())


def _find_booking_journey(reservation: dict, journey_id: str) -> dict | None:
    """Find a booking journey entry within a reservation.

    When multiple bookings share the same journey_id (e.g. after a partial rebook
    creates a 'kept' booking alongside the cancelled original), the first
    non-cancelled booking is returned in preference to a cancelled one.
    """
    first_match = None
    for bk in reservation.get("bookings", []):
        if bk.get("journey_id") == journey_id:
            if first_match is None:
                first_match = bk
            if bk.get("status") != "cancelled":
                return bk
    return first_match


def _find_booking_segment(booking: dict, journey_id: str, flight_number: str = "") -> tuple[list[dict], dict | None]:
    """Find flight segment(s) within a booking journey.

    If flight_number is provided, returns only that segment.
    If omitted and booking has one segment, returns that segment.
    If omitted and booking has multiple segments, returns an error dict.

    Returns:
        (targets, error) — targets is a list of matching segments,
        error is a response dict if something went wrong (else None).
    """
    booking_segments = booking.get("segments", [])
    if flight_number:
        targets = [fs for fs in booking_segments if fs.get("flight_number") == flight_number]
        if not targets:
            return [], {
                "status": "error",
                "error_type": "flight_not_found",
                "message": f"Flight {flight_number} not found in journey {journey_id}",
            }
        return targets, None
    elif len(booking_segments) == 1:
        return booking_segments, None
    else:
        return [], {
            "status": "error",
            "error_type": "flight_number_required",
            "message": f"Journey {journey_id} has {len(booking_segments)} segments; flight_number is required",
        }


def _get_journey_fares(journey: dict) -> dict:
    """Return journey-level fares dict.

    Fares are stored at the journey level as the total price for all segments combined.
    """
    return journey.get("fares", {})


def _get_booking_total_fare(booking: dict) -> float:
    """Return total fare for a booking journey.

    Uses journey-level fare_paid if present, otherwise sums segment-level fare_paid.
    """
    if "fare_paid" in booking:
        return booking.get("fare_paid", 0)
    return sum(seg.get("fare_paid", 0) for seg in booking.get("segments", []))


def _get_journey_available_seats(journey: dict) -> dict:
    """Compute effective available seats for a journey.

    For multi-segment journeys, the constraining factor is the segment
    with the fewest seats in each fare class (min across segments).
    """
    segments = journey.get("segments", [])
    if not segments:
        return {}
    # Start with the first segment's seats
    result = dict(segments[0].get("available_seats", {}))
    # Take the minimum across all segments for each fare class
    for seg in segments[1:]:
        seg_seats = seg.get("available_seats", {})
        for fc in result:
            result[fc] = min(result.get(fc, 0), seg_seats.get(fc, 0))
    return result


def _reservation_not_found(confirmation_number: str):
    return {
        "status": "error",
        "error_type": "not_found",
        "message": f"Reservation {confirmation_number} not found",
    }


def _journey_not_found(journey_id: str):
    return {
        "status": "error",
        "error_type": "journey_not_found",
        "message": f"Journey {journey_id} not found in reservation",
    }


def get_reservation(params: dict, db: dict, call_index: int) -> dict:
    """Retrieve flight reservation details using confirmation number and passenger last name.

    Args:
        params: Tool parameters (confirmation_number, last_name)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with reservation details or error
    """
    try:
        p = GetReservationParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetReservationParams)

    confirmation_number = p.confirmation_number.upper()
    last_name = p.last_name

    # Lookup reservation
    reservation = _lookup_reservation(db, confirmation_number)
    if not reservation:
        return _reservation_not_found(confirmation_number)

    # Validate last_name against reservation passengers
    if last_name:
        passengers = reservation.get("passengers", [])
        last_name_match = any(p2.get("last_name", "").lower() == last_name.lower() for p2 in passengers)
        if not last_name_match:
            return {
                "status": "error",
                "error_type": "authentication_failed",
                "message": f"Last name does not match reservation {confirmation_number}",
            }

    # Write session data to mark successful authentication
    db.setdefault("session", {}).update(
        {
            "confirmation_number": confirmation_number,
            "last_name": last_name.lower() if last_name else "",
        }
    )

    # Return success — sort journeys by first segment's date, then journey_id for readability
    result_reservation = copy.deepcopy(reservation)
    result_reservation["bookings"].sort(
        key=lambda j: (
            j.get("segments", [{}])[0].get("date", ""),
            j.get("journey_id", ""),
        )
    )
    return {"status": "success", "reservation": result_reservation}


def get_flight_status(params: dict, db: dict, call_index: int) -> dict:
    """Get current status of a specific flight including delays, cancellations, and gate information.

    Works for both direct and connecting flights. For connecting flights, you can look up
    by any segment's flight number and it will return the complete flight record.

    Args:
        params: Tool parameters (flight_number, flight_date)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with flight status or error
    """
    try:
        p = GetFlightStatusParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetFlightStatusParams)

    flight_number = p.flight_number.upper()
    flight_date = p.flight_date.replace("-", "")

    # Normalize date format for comparison
    normalized_date = flight_date

    # Lookup flight
    journeys = db.get("journeys", {})

    # Try direct lookup first (works for direct flights and first segment of connecting flights)
    journey_id = f"FL_{flight_number}_{normalized_date}"
    flight = journeys.get(journey_id)

    # If not found, search through all flights for matching segment
    # This handles connecting flights where the flight_number is from a later segment
    if not flight:
        for f in journeys.values():
            # Check if the date matches
            f_date = f.get("date", "").replace("-", "")
            if f_date == normalized_date:
                # Check if any segment has the matching flight number
                for segment in f.get("segments", []):
                    if segment.get("flight_number", "").upper() == flight_number:
                        flight = f
                        break
            if flight:
                break

    # Validate
    if not flight:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"Flight {p.flight_number} not found for date {p.flight_date}",
        }

    # Return success
    return {"status": "success", "journey": copy.deepcopy(flight)}


def get_disruption_info(params: dict, db: dict, call_index: int) -> dict:
    """Get detailed information about a flight disruption for IRROPS handling.

    Args:
        params: Tool parameters (flight_number, date)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with disruption details or error
    """
    try:
        p = GetDisruptionInfoParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, GetDisruptionInfoParams)

    flight_number = p.flight_number.upper()
    date = p.date

    # Look up disruption from dict keyed by "{flight_number}_{date}"
    disruptions = db.get("disruptions", {})
    disruption = None
    if date:
        disruption = disruptions.get(f"{flight_number}_{date}")
    else:
        for d in disruptions.values():
            if d.get("flight_number", "").upper() == flight_number:
                disruption = d
                break

    if not disruption:
        return {
            "status": "error",
            "error_type": "not_found",
            "message": f"No disruption info found for flight {flight_number}",
        }

    return {"status": "success", "disruption": copy.deepcopy(disruption)}


def search_rebooking_options(params: dict, db: dict, call_index: int) -> dict:
    """Search for available flights to rebook a passenger.

    Args:
        params: Tool parameters (origin, destination, date, passenger_count, fare_class)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with list of available flights or error
    """
    try:
        p = SearchRebookingOptionsParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, SearchRebookingOptionsParams)

    origin = p.origin.upper()
    destination = p.destination.upper()
    date = p.date
    passenger_count = p.passenger_count
    fare_class = p.fare_class

    # Get flights
    journeys = db.get("journeys", {})
    results = []

    # Filter flights
    for journey_id, flight in journeys.items():
        # Apply filters
        if flight.get("origin") != origin:
            continue
        if flight.get("destination") != destination:
            continue
        if flight.get("date") != date:
            continue
        if flight.get("status") not in ["scheduled", "on_time", "delayed"]:
            continue
        if not flight.get("bookable", False):
            continue

        # Check seat availability (min across segments for connecting flights)
        available_seats = _get_journey_available_seats(flight)
        if fare_class == "any":
            # Check if ANY cabin class has enough seats
            if not any(seats >= passenger_count for seats in available_seats.values() if seats is not None):
                continue
        else:
            if available_seats.get(fare_class, 0) < passenger_count:
                continue

        # Compute journey-level fares from per-segment fares
        journey_fares = _get_journey_fares(flight)

        # Determine fare_class to use for "any" case
        actual_fare_class = fare_class
        if fare_class == "any":
            # Find cheapest available fare class with enough seats
            available_cabins = [
                (fare_cls, journey_fares.get(fare_cls))
                for fare_cls in ["basic_economy", "main_cabin", "premium_economy", "business", "first"]
                if available_seats.get(fare_cls, 0) >= passenger_count and journey_fares.get(fare_cls) is not None
            ]
            if available_cabins:
                actual_fare_class = min(available_cabins, key=lambda x: x[1])[0]

        # Build result mapping
        segments = flight.get("segments", [])
        all_fare_classes = ["basic_economy", "main_cabin", "premium_economy", "business", "first"]
        result = {
            "journey_id": journey_id,
            "origin": flight.get("origin"),
            "destination": flight.get("destination"),
            "num_stops": flight.get("num_stops", 0),
            "total_duration_minutes": flight.get("total_duration_minutes"),
            "segments": segments,
            "departure_time": segments[0]["scheduled_departure"] if segments else None,
            "arrival_time": segments[-1]["scheduled_arrival"] if segments else None,
            "available_seats": {fc: available_seats.get(fc, 0) for fc in all_fare_classes},
            "fare": journey_fares.get(actual_fare_class),
        }
        results.append(result)

    # Sort by departure time
    results.sort(key=lambda x: x.get("departure_time", ""))

    # Return results
    return {
        "status": "success",
        "options": results,
        "count": len(results),
        "message": f"{len(results)} flight(s) found",
    }


def rebook_flight(params: dict, db: dict, call_index: int) -> dict:
    """Rebook passenger(s) to a new flight.

    Handles voluntary changes, IRROPS rebooking, missed flight recovery,
    and cabin class changes. If new_fare_class is provided, the passenger
    is rebooked into that cabin; otherwise the original fare class is kept.

    When flight_number is provided, performs a partial rebook of just that
    leg within a multi-segment journey (split booking approach): the old
    booking is cancelled, a new booking is created for the replacement leg,
    and another new booking is created for the kept segments.

    Args:
        params: Tool parameters (confirmation_number, journey_id, new_journey_id,
                rebooking_type, waive_change_fee, new_fare_class, flight_number)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with rebooking confirmation and cost summary
    """
    try:
        p = RebookFlightParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, RebookFlightParams)

    confirmation_number = p.confirmation_number.upper()
    journey_id = p.journey_id
    new_journey_id = p.new_journey_id
    rebooking_type = p.rebooking_type
    waive_change_fee = p.waive_change_fee
    new_fare_class = p.new_fare_class
    flight_number = p.flight_number

    # Lookups
    reservation = _lookup_reservation(db, confirmation_number)
    if not reservation:
        return _reservation_not_found(confirmation_number)

    booking = _find_booking_journey(reservation, journey_id)
    if not booking:
        return _journey_not_found(journey_id)

    journeys = db.get("journeys", {})
    new_flight = journeys.get(new_journey_id)
    if not new_flight:
        return {"status": "error", "error_type": "flight_not_found", "message": f"Flight {new_journey_id} not found"}

    if not new_flight.get("bookable", True):
        return {
            "status": "error",
            "error_type": "not_bookable",
            "message": f"Flight {new_journey_id} is not available for booking",
        }

    # Determine fare classes
    original_fare_class = booking.get("fare_class", "main_cabin")
    target_fare_class = new_fare_class or original_fare_class

    # Computations
    is_irrops = "irrops" in rebooking_type

    # Fee calculation — based on ORIGINAL fare class (that's what the passenger paid for)
    fee_map = {"basic_economy": 199, "main_cabin": 75, "premium_economy": 75, "business": 0, "first": 0}
    base_fee = fee_map.get(original_fare_class, 75)
    change_fee = 0 if (is_irrops or waive_change_fee) else base_fee

    # Partial rebook: validate the specific flight segment exists
    if flight_number:
        targets, error = _find_booking_segment(booking, journey_id, flight_number)
        if error:
            return error
        replaced_segment = targets[0]

    # Fare difference — old fare vs new fare in TARGET class
    # For partial rebook, compare only the replaced segment's fare (not the whole booking)
    if flight_number:
        old_fare = replaced_segment.get("fare_paid", 0)
    else:
        old_fare = _get_booking_total_fare(booking)
    journey_fares = _get_journey_fares(new_flight)
    new_fare = journey_fares.get(target_fare_class)

    if new_fare is None:
        return {
            "status": "error",
            "error_type": "fare_class_not_available",
            "message": f"Fare class '{target_fare_class}' is not available on flight {new_journey_id}",
        }

    # Check seat availability in TARGET fare class (min across segments)
    journey_seats = _get_journey_available_seats(new_flight)
    if journey_seats.get(target_fare_class, 0) <= 0:
        return {
            "status": "error",
            "error_type": "no_seats_available",
            "message": f"No seats available in {target_fare_class} on flight {new_journey_id}",
        }

    fare_difference = new_fare - old_fare

    credit_due = max(0, -fare_difference)
    fare_difference_to_collect = max(0, fare_difference)
    total_collected = 0 if is_irrops else (change_fee + fare_difference_to_collect)

    # Mutations
    # Mark old booking as cancelled
    for seg in reservation["bookings"]:
        if seg["journey_id"] == journey_id:
            seg["status"] = "cancelled"
            break

    # Add new booking journey for the replacement flight
    flight_segments = new_flight.get("segments", [])
    old_segments = booking.get("segments", [])
    new_booking_segments = []
    for i, fs in enumerate(flight_segments):
        if is_irrops and not flight_number and i < len(old_segments):
            # Full rebook: preserve old segment fare_paid
            seg_fare_paid = old_segments[i].get("fare_paid", 0)
        elif is_irrops and flight_number and len(flight_segments) == 1:
            # Partial rebook: use replaced_segment's fare_paid
            seg_fare_paid = replaced_segment.get("fare_paid", 0)
        else:
            seg_fare_paid = fs.get("fares", {}).get(target_fare_class, 0)
        new_booking_segments.append(
            {
                "flight_number": fs.get("flight_number"),
                "date": new_flight.get("date"),
                "fare_paid": seg_fare_paid,
                "seat": None,
                "bags_checked": 0,
                "meal_request": None,
            }
        )
    new_booking = {
        "journey_id": new_journey_id,
        "fare_class": target_fare_class,
        "fare_paid": old_fare if is_irrops else new_fare,
        "status": "confirmed",
        "segments": new_booking_segments,
    }
    reservation["bookings"].append(new_booking)

    # Partial rebook: create a kept-segments booking preserving ancillaries
    kept_segments_info = []
    if flight_number:
        kept_segments = [seg for seg in booking.get("segments", []) if seg.get("flight_number") != flight_number]
        if kept_segments:
            kept_booking = {
                "journey_id": journey_id,
                "fare_class": original_fare_class,
                "fare_paid": sum(s.get("fare_paid", 0) for s in kept_segments),
                "status": "confirmed",
                "segments": copy.deepcopy(kept_segments),
            }
            reservation["bookings"].append(kept_booking)
            kept_segments_info = [
                {
                    "flight_number": s.get("flight_number"),
                    "origin": s.get("origin"),
                    "destination": s.get("destination"),
                    "fare_paid": s.get("fare_paid"),
                    "seat": s.get("seat"),
                }
                for s in kept_segments
            ]

    # Update reservation status
    reservation["status"] = "changed"

    # Update seat availability on each segment of the new flight
    for seg in new_flight.get("segments", []):
        seg.setdefault("available_seats", {})
        seg["available_seats"][target_fare_class] = seg["available_seats"].get(target_fare_class, 0) - 1

    # Increment old flight segment seats
    old_journey_id = booking.get("journey_id")
    old_flight = journeys.get(old_journey_id)
    if old_flight:
        for seg in old_flight.get("segments", []):
            seg.setdefault("available_seats", {})
            seg["available_seats"][original_fare_class] = seg["available_seats"].get(original_fare_class, 0) + 1

    # Build response
    response = {
        "status": "success",
        "confirmation_number": confirmation_number,
        "new_journey": {
            "journey_id": new_journey_id,
            "num_stops": new_flight.get("num_stops", 0),
            "segments": copy.deepcopy(new_flight.get("segments", [])),
            "departure": new_flight["segments"][0]["scheduled_departure"] if new_flight.get("segments") else None,
            "arrival": new_flight["segments"][-1]["scheduled_arrival"] if new_flight.get("segments") else None,
            "origin": new_flight.get("origin"),
            "destination": new_flight.get("destination"),
        },
        "cost_summary": {
            "original_fare_class": original_fare_class,
            "new_fare_class": target_fare_class,
            "cabin_changed": original_fare_class != target_fare_class,
            "change_fee": change_fee,
            "fare_difference": fare_difference,
            "credit_due": credit_due,
            "total_collected": total_collected,
            "fee_waived": is_irrops or waive_change_fee,
        },
        "message": f"Successfully rebooked to flight {new_journey_id}"
        + (f" in {target_fare_class}" if original_fare_class != target_fare_class else ""),
    }

    if flight_number:
        response["partial_rebook"] = True
        response["replaced_segment"] = {
            "flight_number": flight_number,
            "origin": replaced_segment.get("origin"),
            "destination": replaced_segment.get("destination"),
            "fare_paid": replaced_segment.get("fare_paid"),
        }
        response["kept_segments"] = kept_segments_info

    return response


def add_to_standby(params: dict, db: dict, call_index: int) -> dict:
    """Add passenger to standby list for a flight.

    Args:
        params: Tool parameters (confirmation_number, journey_id, passenger_ids)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with standby confirmation
    """
    try:
        p = AddToStandbyParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, AddToStandbyParams)

    confirmation_number = p.confirmation_number.upper()
    journey_id = p.journey_id
    passenger_ids = p.passenger_ids

    # Lookup flight
    journeys = db.get("journeys", {})
    flight = journeys.get(journey_id)

    # Validate
    if not flight:
        return {"status": "error", "error_type": "flight_not_found", "message": f"Flight {journey_id} not found"}

    if flight.get("status") == "cancelled":
        return {
            "status": "error",
            "error_type": "flight_cancelled",
            "message": "Cannot add to standby for cancelled flight",
        }

    # Validate passenger_ids against reservation
    reservation = _lookup_reservation(db, confirmation_number)
    if reservation and passenger_ids:
        valid_passenger_ids = {p2.get("passenger_id") for p2 in reservation.get("passengers", [])}
        invalid_ids = [pid for pid in passenger_ids if pid not in valid_passenger_ids]
        if invalid_ids:
            return {
                "status": "error",
                "error_type": "invalid_passengers",
                "message": f"Unknown passenger ID(s): {', '.join(invalid_ids)}",
            }

    # Mutations
    # Add standby list to flight if it doesn't exist
    if "standby_list" not in flight:
        flight["standby_list"] = []

    # Position of the last added passenger in the queue after insertion
    standby_position = len(flight["standby_list"]) + len(passenger_ids)

    if reservation:
        # Initialize standby_list if it doesn't exist
        if "standby_list" not in reservation:
            reservation["standby_list"] = []

        # Add standby entry for this flight
        reservation["standby_list"].append(
            {
                "journey_id": journey_id,
                "passenger_ids": passenger_ids,
                "position": standby_position,
                "status": "pending",
            }
        )

    # Add passengers to flight's standby list
    for passenger_id in passenger_ids:
        flight["standby_list"].append(
            {
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "position": len(flight["standby_list"]) + 1,
            }
        )

    # Return success
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "journey_id": journey_id,
        "standby_list_position": standby_position,
        "message": f"Added {len(passenger_ids)} passenger(s) to standby list",
    }


def assign_seat(params: dict, db: dict, call_index: int) -> dict:
    """Assign a seat to a passenger on a specific flight segment.

    Args:
        params: Tool parameters (confirmation_number, passenger_id, journey_id,
                seat_preference, flight_number)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with seat assignment details
    """
    try:
        p = AssignSeatParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, AssignSeatParams)

    confirmation_number = p.confirmation_number.upper()
    passenger_id = p.passenger_id
    journey_id = p.journey_id
    seat_preference = p.seat_preference
    flight_number = p.flight_number

    # Lookups
    reservation = _lookup_reservation(db, confirmation_number)
    if not reservation:
        return _reservation_not_found(confirmation_number)

    booking = _find_booking_journey(reservation, journey_id)
    if not booking:
        return _journey_not_found(journey_id)

    targets, error = _find_booking_segment(booking, journey_id, flight_number)
    if error:
        return error
    flight_seg = targets[0]
    if not flight_number:
        flight_number = flight_seg.get("flight_number", "")

    # Find the corresponding journey segment for availability checks
    journeys = db.get("journeys", {})
    journey = journeys.get(journey_id)
    journey_seg = None
    if journey:
        for js in journey.get("segments", []):
            if js.get("flight_number") == flight_number:
                journey_seg = js
                break

    # Validate flight availability
    fare_class = booking.get("fare_class", "main_cabin")
    if journey_seg:
        seg_seats = journey_seg.get("available_seats", {}).get(fare_class, 0)
        if seg_seats <= 0:
            return {
                "status": "error",
                "error_type": "no_seats_available",
                "message": f"No seats available in {fare_class} fare class",
            }

        # Check seat type availability for this fare class
        raw_seat_types = journey_seg.get("available_seat_types")
        if raw_seat_types and isinstance(raw_seat_types, dict):
            available_seat_types = raw_seat_types.get(fare_class, ["window", "aisle", "middle"])
        else:
            available_seat_types = ["window", "aisle", "middle"]
        if seat_preference != "no_preference" and seat_preference not in available_seat_types:
            return {
                "status": "error",
                "error_type": "seat_type_unavailable",
                "message": f"No {seat_preference} seats available in {fare_class} on this flight. Available types: {', '.join(available_seat_types)}",
            }

    # Compute seat number
    passenger_index = int(passenger_id[-3:]) if passenger_id and len(passenger_id) >= 3 else 0
    base_row_map = {"basic_economy": 25, "main_cabin": 20, "premium_economy": 10, "business": 5, "first": 1}
    base_row = base_row_map.get(fare_class, 20)
    seat_row = base_row + passenger_index

    seat_letter_map = {"window": "A", "aisle": "C", "middle": "B", "no_preference": "C"}
    seat_letter = seat_letter_map.get(seat_preference, "C")
    seat_number = f"{seat_row}{seat_letter}"

    # Mutations — update seat on the specific flight segment
    flight_seg["seat"] = seat_number

    # Note: We do NOT decrement available_seats here because the passenger
    # already has a confirmed booking (seat was claimed during booking/rebooking).
    # This tool just assigns a specific seat number, not a new seat.

    # Return response
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "journey_id": journey_id,
        "flight_number": flight_number,
        "seat_assigned": seat_number,
        "fare_class": fare_class,
        "preference": seat_preference,
        "message": f"Seat {seat_number} ({seat_preference}) successfully assigned",
    }


def add_baggage_allowance(params: dict, db: dict, call_index: int) -> dict:
    """Add checked baggage allowance to a flight segment within a booking journey.

    Args:
        params: Tool parameters (confirmation_number, journey_id, num_bags, flight_number)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with baggage allowance confirmation
    """
    try:
        p = AddBaggageAllowanceParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, AddBaggageAllowanceParams)

    confirmation_number = p.confirmation_number.upper()
    journey_id = p.journey_id
    num_bags = p.num_bags
    flight_number = p.flight_number

    # Lookups
    reservation = _lookup_reservation(db, confirmation_number)
    if not reservation:
        return _reservation_not_found(confirmation_number)

    booking = _find_booking_journey(reservation, journey_id)
    if not booking:
        return _journey_not_found(journey_id)

    # Validate
    if num_bags < 0 or num_bags > 5:
        return {
            "status": "error",
            "error_type": "invalid_bag_count",
            "message": f"Invalid number of bags {num_bags}. Must be between 0 and 5",
        }

    # Find target flight segment(s) — if no flight_number, apply to all
    if flight_number:
        targets, error = _find_booking_segment(booking, journey_id, flight_number)
        if error:
            return error
    else:
        targets = booking.get("segments", [])

    # Mutations — set baggage count on target segment(s)
    for fs in targets:
        fs["bags_checked"] = num_bags

    # Return response
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "journey_id": journey_id,
        "bags_checked": num_bags,
        "message": f"Baggage allowance set to {num_bags} checked bag(s)",
    }


def add_meal_request(params: dict, db: dict, call_index: int) -> dict:
    """Add or update special meal request for a passenger on a flight segment.

    Args:
        params: Tool parameters (confirmation_number, passenger_id, journey_id,
                meal_type, flight_number)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with meal request confirmation
    """
    try:
        p = AddMealRequestParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, AddMealRequestParams)

    confirmation_number = p.confirmation_number.upper()
    passenger_id = p.passenger_id
    journey_id = p.journey_id
    meal_type = p.meal_type
    flight_number = p.flight_number

    # Lookups
    reservation = _lookup_reservation(db, confirmation_number)
    if not reservation:
        return _reservation_not_found(confirmation_number)

    booking = _find_booking_journey(reservation, journey_id)
    if not booking:
        return _journey_not_found(journey_id)

    # Find target flight segment(s) — if no flight_number, apply to all
    if flight_number:
        targets, error = _find_booking_segment(booking, journey_id, flight_number)
        if error:
            return error
    else:
        targets = booking.get("segments", [])

    # Mutations — update meal request on target segment(s)
    for fs in targets:
        fs["meal_request"] = meal_type

    # Return response
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "journey_id": journey_id,
        "meal_type": meal_type,
        "message": f"{meal_type} meal request added",
    }


def issue_travel_credit(params: dict, db: dict, call_index: int) -> dict:
    """Issue a travel credit or future flight voucher to the passenger.

    Args:
        params: Tool parameters (confirmation_number, passenger_id, amount, credit_reason)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with credit details
    """
    try:
        p = IssueTravelCreditParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, IssueTravelCreditParams)

    confirmation_number = p.confirmation_number.upper()
    passenger_id = p.passenger_id
    amount = p.amount
    credit_reason = p.credit_reason

    # Validate reservation exists
    if not _lookup_reservation(db, confirmation_number):
        return _reservation_not_found(confirmation_number)

    # Compute credit code
    passenger_prefix = passenger_id[:3].upper() if passenger_id else ""
    credit_code = f"TC{confirmation_number}{passenger_prefix}"

    # Mutations (if enabled)
    # Add travel credit to database
    travel_credits = db.setdefault("travel_credits", {})
    travel_credits[credit_code] = {
        "credit_code": credit_code,
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "amount": amount,
        "credit_reason": credit_reason,
        "issued_date": db["_current_date"],
        "expiry_date": str(int(db["_current_date"][:4]) + 1) + db["_current_date"][4:],
        "status": "active",
    }

    # Return response
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "credit_code": credit_code,
        "amount": amount,
        "valid_months": 12,
        "message": f"${amount} travel credit issued with code {credit_code}",
    }


def issue_hotel_voucher(params: dict, db: dict, call_index: int) -> dict:
    """Issue a hotel voucher.

    Args:
        params: Tool parameters (confirmation_number, passenger_id, num_nights)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with voucher details
    """
    try:
        p = IssueHotelVoucherParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, IssueHotelVoucherParams)

    confirmation_number = p.confirmation_number.upper()
    passenger_id = p.passenger_id
    num_nights = p.num_nights

    # Validate reservation exists
    if not _lookup_reservation(db, confirmation_number):
        return _reservation_not_found(confirmation_number)

    # Validate
    if num_nights > 3:
        return {
            "status": "error",
            "error_type": "exceeds_authority",
            "message": "Hotel vouchers can be issued for maximum of 3 nights",
        }

    # Compute voucher code
    voucher_code = f"HOTEL-{confirmation_number}"

    # Mutations (if enabled)
    # Add hotel voucher to database
    hotel_vouchers = db.setdefault("hotel_vouchers", {})
    hotel_vouchers[voucher_code] = {
        "voucher_code": voucher_code,
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "num_nights": num_nights,
        "issued_date": db["_current_date"],
        "status": "active",
    }

    # Return response
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "voucher_code": voucher_code,
        "number_of_nights": num_nights,
        "valid_at": "Any hotels in airport area",
        "message": f"Hotel voucher issued with code {voucher_code} for {num_nights} nights",
    }


def issue_meal_voucher(params: dict, db: dict, call_index: int) -> dict:
    """Issue a meal voucher for delays or disruptions that qualify.

    Args:
        params: Tool parameters (confirmation_number, passenger_id, voucher_reason)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with voucher details
    """
    try:
        p = IssueMealVoucherParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, IssueMealVoucherParams)

    confirmation_number = p.confirmation_number.upper()
    passenger_id = p.passenger_id
    voucher_reason = p.voucher_reason

    # Validate reservation exists
    if not _lookup_reservation(db, confirmation_number):
        return _reservation_not_found(confirmation_number)

    # Compute amount based on reason
    amount_map = {
        "delay_over_2_hours": 12,
        "delay_over_4_hours": 15,
        "cancellation_wait_same_day": 15,
        "irrops_overnight": 25,
    }
    amount = amount_map.get(voucher_reason, 12)

    # Compute voucher code
    passenger_prefix = passenger_id[:4].upper() if passenger_id else ""
    voucher_code = f"MEAL-{confirmation_number}-{passenger_prefix}"

    # Mutations (if enabled)
    # Add meal voucher to database
    meal_vouchers = db.setdefault("meal_vouchers", {})
    meal_vouchers[voucher_code] = {
        "voucher_code": voucher_code,
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "amount": amount,
        "voucher_reason": voucher_reason,
        "issued_date": db["_current_date"],
        "status": "active",
    }

    # Return response
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "passenger_id": passenger_id,
        "voucher_code": voucher_code,
        "amount": amount,
        "valid_at": "Airport terminal restaurants",
        "message": f"${amount} meal voucher issued with code {voucher_code}",
    }


def cancel_reservation(params: dict, db: dict, call_index: int) -> dict:
    """Cancel a specific booking journey in a reservation.

    If all journeys end up cancelled, the reservation itself is marked cancelled.

    Args:
        params: Tool parameters (confirmation_number, journey_id, cancellation_reason)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with cancellation details for the journey
    """
    try:
        p = CancelReservationParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, CancelReservationParams)

    confirmation_number = p.confirmation_number.upper()
    journey_id = p.journey_id
    cancellation_reason = p.cancellation_reason

    # Lookups
    reservation = _lookup_reservation(db, confirmation_number)
    if not reservation:
        return _reservation_not_found(confirmation_number)

    booking = _find_booking_journey(reservation, journey_id)
    if not booking:
        return _journey_not_found(journey_id)

    if booking.get("status") == "cancelled":
        return {
            "status": "error",
            "error_type": "already_cancelled",
            "message": f"Journey {journey_id} is already cancelled",
        }

    # Compute refund/credit for this booking journey's fare
    booking_fare = _get_booking_total_fare(booking)

    is_refundable = (
        "irrops_refund" in cancellation_reason
        or "24_hour_rule" in cancellation_reason
        or reservation.get("fare_type") == "refundable"
    )

    cancellation_fee = (
        0
        if (
            "24_hour_rule" in cancellation_reason
            or "irrops_refund" in cancellation_reason
            or reservation.get("fare_type") == "refundable"
        )
        else 100
    )

    refund_amount = max(0, booking_fare - cancellation_fee) if is_refundable else 0
    credit_amount = 0 if is_refundable else max(0, booking_fare - cancellation_fee)

    # Mutate: cancel the booking journey
    booking["status"] = "cancelled"

    # Increment seat availability on cancelled journey segments
    fare_class = booking.get("fare_class", "main_cabin")
    journeys = db.get("journeys", {})
    cancelled_flight = journeys.get(journey_id)
    if cancelled_flight:
        for seg in cancelled_flight.get("segments", []):
            seg.setdefault("available_seats", {})
            seg["available_seats"][fare_class] = seg["available_seats"].get(fare_class, 0) + 1

    # Check if all segments are now cancelled
    all_cancelled = all(seg.get("status") == "cancelled" for seg in reservation.get("bookings", []))
    if all_cancelled:
        reservation["status"] = "cancelled"

    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "journey_id": journey_id,
        "is_refundable": is_refundable,
        "cancellation_fee": cancellation_fee,
        "refund_amount_eligible": refund_amount,
        "credit_amount_eligible": credit_amount,
        "reservation_status": "cancelled" if all_cancelled else "active",
        "message": f"Journey {journey_id} cancelled successfully",
    }


def process_refund(params: dict, db: dict, call_index: int) -> dict:
    """Process a refund for a cancelled or eligible reservation.

    Args:
        params: Tool parameters (confirmation_number, refund_amount, refund_type)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with refund details
    """
    try:
        p = ProcessRefundParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, ProcessRefundParams)

    confirmation_number = p.confirmation_number.upper()
    refund_amount = p.refund_amount
    refund_type = p.refund_type

    # Validate reservation exists
    if not _lookup_reservation(db, confirmation_number):
        return _reservation_not_found(confirmation_number)

    # Validate refund_amount
    if refund_amount <= 0:
        return {
            "status": "error",
            "error_type": "invalid_amount",
            "message": f"Refund amount must be greater than 0, got {refund_amount}",
        }

    # Compute refund ID
    refund_id = f"REF-{confirmation_number}-{str(call_index).zfill(3)}"
    processing_days = 7

    # Mutations (if enabled)
    # Add refund to database
    refunds = db.setdefault("refunds", {})
    refunds[refund_id] = {
        "refund_id": refund_id,
        "confirmation_number": confirmation_number,
        "refund_amount": refund_amount,
        "refund_type": refund_type,
        "processing_days": processing_days,
        "initiated_date": db["_current_date"],
        "status": "processing",
    }

    # Return response
    return {
        "status": "success",
        "confirmation_number": confirmation_number,
        "refund_id": refund_id,
        "refund_amount": refund_amount,
        "refund_type": refund_type,
        "processing_days": processing_days,
        "message": f"${refund_amount} refund initiated, processing time {processing_days} business days",
    }


def transfer_to_agent(params: dict, db: dict, call_index: int) -> dict:
    """Transfer the call to a live human agent.

    Args:
        params: Tool parameters (confirmation_number, transfer_reason, issue_summary)
        db: Scenario database
        call_index: Call count for this tool

    Returns:
        Response with transfer details
    """
    try:
        p = TransferToAgentParams.model_validate(params)
    except ValidationError as exc:
        return validation_error_response(exc, TransferToAgentParams)

    confirmation_number = p.confirmation_number.upper()
    transfer_reason = p.transfer_reason
    issue_summary = p.issue_summary

    # Validate reservation exists
    if confirmation_number and not _lookup_reservation(db, confirmation_number):
        return _reservation_not_found(confirmation_number)

    # Compute transfer ID
    transfer_id = f"TRF-{confirmation_number}-{str(call_index).zfill(3)}"

    # Return response
    return {
        "status": "success",
        "transfer_id": transfer_id,
        "confirmation_number": confirmation_number,
        "transfer_reason": transfer_reason,
        "issue_summary": issue_summary,
        "estimated_wait": "2-3 minutes",
        "message": "Transferring to live agent",
    }
