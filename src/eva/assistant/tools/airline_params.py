"""Pydantic parameter models and enums for airline tool functions.

Each tool function has a corresponding *Params model that validates and types
its incoming ``params`` dict. Call ``Model.model_validate(params)`` at the top
of the tool function and catch ``ValidationError`` to produce a standard
``{"status": "error", ...}`` response for bad LLM-supplied inputs.

Enums use ``StrEnum`` so values serialise to plain strings in JSON and
compare equal to their string counterparts (e.g. ``FareClass.main_cabin == "main_cabin"``).
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError


class FareClass(StrEnum):
    basic_economy = "basic_economy"
    main_cabin = "main_cabin"
    premium_economy = "premium_economy"
    business = "business"
    first = "first"


class FareClassOrAny(StrEnum):
    """Fare class for search queries; ``any`` means cheapest available."""

    basic_economy = "basic_economy"
    main_cabin = "main_cabin"
    premium_economy = "premium_economy"
    business = "business"
    first = "first"
    any = "any"


class RebookingType(StrEnum):
    voluntary = "voluntary"
    same_day = "same_day"
    irrops_cancellation = "irrops_cancellation"
    irrops_delay = "irrops_delay"
    irrops_schedule_change = "irrops_schedule_change"
    missed_flight_passenger_fault = "missed_flight_passenger_fault"
    missed_connection_airline_fault = "missed_connection_airline_fault"


class SeatPreference(StrEnum):
    window = "window"
    aisle = "aisle"
    middle = "middle"
    no_preference = "no_preference"


class MealType(StrEnum):
    vegetarian = "vegetarian"
    vegan = "vegan"
    kosher = "kosher"
    halal = "halal"
    gluten_free = "gluten_free"
    diabetic = "diabetic"
    low_sodium = "low_sodium"
    child = "child"
    hindu = "hindu"
    none = "none"
    standard = "standard"


class CreditReason(StrEnum):
    cancellation_non_refundable = "cancellation_non_refundable"
    fare_difference_negative = "fare_difference_negative"
    service_recovery = "service_recovery"
    goodwill = "goodwill"
    downgrade_compensation = "downgrade_compensation"


class VoucherReason(StrEnum):
    delay_over_2_hours = "delay_over_2_hours"
    delay_over_4_hours = "delay_over_4_hours"
    cancellation_wait_same_day = "cancellation_wait_same_day"
    irrops_overnight = "irrops_overnight"


class RefundType(StrEnum):
    full_fare = "full_fare"
    partial_fare = "partial_fare"
    taxes_only = "taxes_only"
    ancillary_fees = "ancillary_fees"


ConfirmationNumber = Annotated[str, Field(pattern=r"^[A-Za-z0-9]{6}$", description="6 alphanumeric characters")]
FlightNumberStr = Annotated[
    str, Field(pattern=r"^[A-Za-z]{2,3}\d{1,4}$", description="2-3 letters followed by 1-4 digits", examples=["SK621"])
]
DateStr = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")]
JourneyIdStr = Annotated[
    str,
    Field(
        pattern=r"^FL_(?:[A-Za-z]{2,3}\d{1,4}_)+\d{8}$",
        description="FL_<flight>_<YYYYMMDD>",
        examples=["FL_SK621_20260320"],
    ),
]
PassengerIdStr = Annotated[str, Field(pattern=r"^PAX\d+$", description="PAX<digits>", examples=["PAX001"])]
AirportCode = Annotated[str, Field(pattern=r"^[A-Za-z]{3}$", description="3-letter airport code", examples=["JFK"])]


class GetReservationParams(BaseModel):
    confirmation_number: ConfirmationNumber
    last_name: str


class GetFlightStatusParams(BaseModel):
    flight_number: FlightNumberStr
    flight_date: DateStr


class GetDisruptionInfoParams(BaseModel):
    flight_number: FlightNumberStr
    date: DateStr


class SearchRebookingOptionsParams(BaseModel):
    origin: AirportCode
    destination: AirportCode
    date: DateStr
    passenger_count: int = Field(ge=1)
    fare_class: FareClassOrAny


class RebookFlightParams(BaseModel):
    confirmation_number: ConfirmationNumber
    journey_id: JourneyIdStr
    new_journey_id: JourneyIdStr
    rebooking_type: RebookingType
    waive_change_fee: bool
    new_fare_class: FareClass | None = None
    flight_number: FlightNumberStr | None = None


class AddToStandbyParams(BaseModel):
    confirmation_number: ConfirmationNumber
    journey_id: JourneyIdStr
    passenger_ids: list[PassengerIdStr]


class AssignSeatParams(BaseModel):
    confirmation_number: ConfirmationNumber
    passenger_id: PassengerIdStr
    journey_id: JourneyIdStr
    seat_preference: SeatPreference
    flight_number: FlightNumberStr | None = None


class AddBaggageAllowanceParams(BaseModel):
    confirmation_number: ConfirmationNumber
    journey_id: JourneyIdStr
    num_bags: int = Field(ge=0, le=5)
    flight_number: FlightNumberStr | None = None


class AddMealRequestParams(BaseModel):
    confirmation_number: ConfirmationNumber
    passenger_id: PassengerIdStr
    journey_id: JourneyIdStr
    meal_type: MealType
    flight_number: FlightNumberStr | None = None


class IssueTravelCreditParams(BaseModel):
    confirmation_number: ConfirmationNumber
    passenger_id: PassengerIdStr
    amount: float = Field(gt=0)
    credit_reason: CreditReason


class IssueHotelVoucherParams(BaseModel):
    confirmation_number: ConfirmationNumber
    passenger_id: PassengerIdStr
    num_nights: int = Field(ge=1)


class IssueMealVoucherParams(BaseModel):
    confirmation_number: ConfirmationNumber
    passenger_id: PassengerIdStr
    voucher_reason: VoucherReason


class CancelReservationParams(BaseModel):
    confirmation_number: ConfirmationNumber
    journey_id: JourneyIdStr
    cancellation_reason: str


class ProcessRefundParams(BaseModel):
    confirmation_number: ConfirmationNumber
    refund_amount: float = Field(gt=0)
    refund_type: RefundType


class TransferToAgentParams(BaseModel):
    confirmation_number: ConfirmationNumber
    transfer_reason: str
    issue_summary: str


# Maps Pydantic field names → (error_type, human-readable label)
FIELD_ERROR_TYPES: dict[str, tuple[str, str]] = {
    # Enum fields
    "rebooking_type": ("invalid_rebooking_type", "rebooking_type"),
    "new_fare_class": ("invalid_fare_class", "new_fare_class"),
    "fare_class": ("invalid_fare_class", "fare_class"),
    "meal_type": ("invalid_meal_type", "meal_type"),
    "credit_reason": ("invalid_credit_reason", "credit_reason"),
    "voucher_reason": ("invalid_voucher_reason", "voucher_reason"),
    "refund_type": ("invalid_refund_type", "refund_type"),
    "seat_preference": ("invalid_seat_preference", "seat_preference"),
    # Format-validated fields
    "confirmation_number": ("invalid_confirmation_number_format", "confirmation_number"),
    "flight_number": ("invalid_flight_number_format", "flight_number"),
    "flight_date": ("invalid_date_format", "flight_date"),
    "date": ("invalid_date_format", "date"),
    "journey_id": ("invalid_journey_id_format", "journey_id"),
    "new_journey_id": ("invalid_journey_id_format", "new_journey_id"),
    "passenger_id": ("invalid_passenger_id_format", "passenger_id"),
    "passenger_ids": ("invalid_passenger_id_format", "passenger_ids"),
    "origin": ("invalid_airport_code_format", "origin"),
    "destination": ("invalid_airport_code_format", "destination"),
    "num_bags": ("invalid_bag_count", "num_bags"),
}


def validation_error_response(exc: ValidationError, model: type[BaseModel]) -> dict:
    """Convert a Pydantic ``ValidationError`` to a standard tool error response.

    Produces ``{"status": "error", "error_type": ..., "message": ...}`` matching
    the format returned by inline validation in each tool function.

    Uses ``model.model_fields`` to pull ``description`` and ``examples`` from
    the field's ``Field(...)`` metadata for human-friendly messages.
    """
    for error in exc.errors():
        loc = error.get("loc", ())
        if loc:
            field = str(loc[0])
            if field in FIELD_ERROR_TYPES:
                error_type, label = FIELD_ERROR_TYPES[field]
                input_val = error.get("input", "")
                msg = f"Invalid {label} '{input_val}'"
                if (field_info := model.model_fields.get(field)) and field_info.description:
                    msg += f": must be {field_info.description}"
                    if field_info.examples:
                        msg += f" (e.g. {', '.join(str(e) for e in field_info.examples)})"
                elif detail := error.get("msg", ""):
                    msg += f": {detail}"
                return {
                    "status": "error",
                    "error_type": error_type,
                    "message": msg,
                }
    # Generic fallback for missing required fields or other validation errors
    first = exc.errors()[0] if exc.errors() else {}
    loc = first.get("loc", ("parameter",))
    field = str(loc[0]) if loc else "parameter"
    return {
        "status": "error",
        "error_type": "invalid_parameter",
        "message": f"Invalid or missing parameter '{field}': {first.get('msg', str(exc))}",
    }
