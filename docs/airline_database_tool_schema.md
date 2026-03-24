# Airline Database & Tool Schema

## Database Schema

### Reservations Table
Keyed by `confirmation_number`. Represents a booking.

**Top-level fields:**
| Field | Type | Notes |
|---|---|---|
| `confirmation_number` | str | Primary key |
| `status` | str | `"confirmed"` \| `"changed"` \| `"cancelled"` |
| `booking_date` | str | ISO datetime |
| `fare_type` | str | `"refundable"` \| `"non_refundable"` |
| `passengers` | list | See Passenger below |
| `bookings` | list | See Booking Journey below |
| `ancillaries` | dict | `seat_selection_fee`, `bags_fee` |

**Passenger:**
| Field | Type | Notes |
|---|---|---|
| `passenger_id` | str | e.g. `"PAX001"` |
| `first_name` | str | |
| `last_name` | str | |
| `ticket_number` | str | |
| `email` | str | |
| `phone` | str | |
| `elite_status` | str \| None | |
| `meal_preference` | str \| None | Stored preference; not auto-applied by tools |
| `seat_preference` | str \| None | Stored preference; not auto-applied by tools |

**Booking Journey** (one journey booked under this reservation):
| Field | Type | Notes |
|---|---|---|
| `journey_id` | str | References journeys table |
| `fare_class` | str | See fare_class enum |
| `fare_paid` | float | Total paid for this journey (all legs) |
| `status` | str | `"confirmed"` \| `"cancelled"` |
| `segments` | list | See Booking Segment below |

**Booking Segment** (one flight leg within a booking journey):
| Field | Type | Notes |
|---|---|---|
| `flight_number` | str | |
| `date` | str | `"YYYY-MM-DD"` |
| `fare_paid` | float | Amount paid for this specific leg |
| `seat` | str \| None | e.g. `"22C"`; None until assigned |
| `bags_checked` | int | |
| `meal_request` | str \| None | |

> **Note:** `origin`, `destination`, `departure_time`, and `arrival_time` are NOT
> stored on booking segments. Look those up in the journeys table by `journey_id`.

---

### Journeys Table
Keyed by `journey_id`. Represents an available flight or connecting itinerary in the system.

**Journey ID format:** `FL_{flight_number}_{YYYYMMDD}` for direct flights;
`FL_{fn1}_{fn2}_{YYYYMMDD}` for connecting flights.

**Top-level fields:**
| Field | Type | Notes |
|---|---|---|
| `journey_id` | str | Primary key |
| `date` | str | `"YYYY-MM-DD"` |
| `origin` | str | IATA airport code |
| `destination` | str | IATA airport code |
| `num_stops` | int | 0 = direct |
| `total_duration_minutes` | int | |
| `status` | str | `"scheduled"` \| `"on_time"` \| `"delayed"` \| `"cancelled"` \| `"departed"` |
| `bookable` | bool | False if cancelled or otherwise unavailable for booking |
| `fares` | dict | Total price per cabin class for the whole journey (see note below) |
| `segments` | list | See Journey Segment below |

**`fares` dict** — keyed by fare class, value is the total journey price (all legs combined).
`None` means that class is unavailable on this journey.
```json
{
  "basic_economy": 280.0,
  "main_cabin": 400.0,
  "premium_economy": 600.0,
  "business": 1300.0,
  "first": 2600.0
}
```

**Journey Segment** (one flight leg within a journey):
| Field | Type | Notes |
|---|---|---|
| `segment_number` | int | 1-indexed |
| `flight_number` | str | |
| `origin` | str | IATA code |
| `destination` | str | IATA code |
| `scheduled_departure` | str | Local time `"HH:MM"` |
| `scheduled_arrival` | str | Local time `"HH:MM"` |
| `origin_utc_offset` | int | |
| `destination_utc_offset` | int | |
| `duration_minutes` | int | |
| `aircraft_type` | str | |
| `status` | str | `"scheduled"` \| `"on_time"` \| `"delayed"` \| `"cancelled"` \| `"departed"` |
| `delay_minutes` | int \| None | |
| `delay_reason` | str \| None | |
| `cancellation_reason` | str \| None | |
| `gate` | str \| None | |
| `available_seats` | dict | Available seats by fare class (int); see fare_class enum for keys |
| `available_seat_types` | dict | Available seats types by fare class (list of either middle, aisle, window); see fare_class enum for keys |
| `fares` | dict | Per-leg price by fare class; used for partial rebook pricing |

---

### Disruptions Table
Dict of active disruptions keyed by `"{flight_number}_{date}"` (e.g. `"SW100_2026-03-20"`).
One entry per disrupted flight. `get_disruption_info` does a direct O(1) key lookup when
`date` is provided; otherwise scans values by `flight_number`.

| Field | Type | Notes |
|---|---|---|
| `flight_number` | str | |
| `date` | str | `"YYYY-MM-DD"` |
| `disruption_type` | str | `"cancellation"` \| `"delay"` \| `"schedule_change"` |
| `cause` | str | Human-readable cause |
| `cause_category` | str | `"weather"` \| `"airline_fault"` \| `"mechanical"` |
| `is_irrops` | bool | True = irregular operations (triggers fee waivers) |
| `delay_minutes` | int \| None | |
| `passenger_entitled_to` | dict | See below |

**`passenger_entitled_to`:**
| Field | Type |
|---|---|
| `fee_waiver` | bool |
| `refund_option` | bool |
| `meal_voucher` | bool |
| `hotel_accommodation` | bool |
| `rebooking_window_days` | int |

---

### Other Tables (created by write tools)
| Table | Key | Created by |
|---|---|---|
| `travel_credits` | `credit_code` | `issue_travel_credit` |
| `meal_vouchers` | `voucher_code` | `issue_meal_voucher` |
| `hotel_vouchers` | `voucher_code` | `issue_hotel_voucher` |
| `refunds` | `refund_id` | `process_refund` |

---

## Tools

### Read Tools

| Tool | Parameters | Validation | Queries | Returns |
|---|---|---|---|---|
| `get_reservation` | `confirmation_number` (str, required) <br>`last_name` (str, optional) | If `last_name` provided, must match a passenger in the reservation | Reservations table by `confirmation_number.upper()` | Full reservation dict; journeys sorted by (segment date, journey_id) |
| `get_flight_status` | `flight_number` (str, required) <br>`flight_date` (str, required, `"YYYY-MM-DD"`) | None | Journeys table: tries key `FL_{flight_number}_{date}` first; if not found, scans all journeys on that date for a segment with that flight_number | Full journey dict including all segments with status, gate, delay info |
| `get_disruption_info` | `flight_number` (str, required) <br>`date` (str, optional, `"YYYY-MM-DD"`) | None | Disruptions list: scans for matching flight_number (and date if provided) | Full disruption dict including `passenger_entitled_to` entitlements |
| `search_rebooking_options` | `origin` (str, required) <br>`destination` (str, required) <br>`date` (str, required, `"YYYY-MM-DD"`) <br>`passenger_count` (int, default 1) <br>`fare_class` (str, default `"main_cabin"`; `"any"` returns cheapest available) | None | Journeys table: filters by origin, destination, date, status in (scheduled/on_time/delayed), bookable=true, available seats ≥ passenger_count in fare class | List of options with: `journey_id`, `departure_time`, `arrival_time`, `num_stops`, `segments`, `available_seats` (per class), `available_seat_types`, `fare` (for requested class), `count` |

---

### Write Tools

| Tool | Parameters | Validation | Queries | Modifies | Returns |
|---|---|---|---|---|---|
| `rebook_flight` | `confirmation_number` (str, required) <br>`journey_id` (str, required — booking journey to replace) <br>`new_journey_id` (str, required) <br>`rebooking_type` (str, required) <br>`waive_change_fee` (bool, default false) <br>`new_fare_class` (str, optional — defaults to original fare class) <br>`flight_number` (str, optional — triggers partial rebook of one leg within a multi-segment journey) | `rebooking_type` must be valid; `new_fare_class` must be valid if provided; reservation + journey must exist; new journey must exist and be bookable; target fare class must have available seats | Reservations (by confirmation_number), Journeys (by new_journey_id) | Old booking journey status → `"cancelled"`; new booking journey appended to reservation; for partial rebook: old cancelled + new leg booking + kept-segments booking all appended; available seats decremented on new journey segments; reservation status → `"changed"` | `confirmation_number`, `new_journey`, `cost_summary` (`change_fee`, `fare_difference`, `credit_due`, `total_collected`, `fee_waived`, `cabin_changed`); if partial rebook: `partial_rebook=True`, `replaced_segment`, `kept_segments` |
| `cancel_reservation` | `confirmation_number` (str, required) <br>`journey_id` (str, required) <br>`cancellation_reason` (str, required) | Reservation + journey must exist; journey must not already be cancelled | Reservations | Booking journey status → `"cancelled"`; if ALL journeys now cancelled, reservation status → `"cancelled"` | `journey_id`, `is_refundable`, `cancellation_fee`, `refund_amount`, `credit_amount`, `reservation_status` |
| `assign_seat` | `confirmation_number` (str, required) <br>`passenger_id` (str, required) <br>`journey_id` (str, required) <br>`seat_preference` (str, default `"no_preference"`) <br>`flight_number` (str, optional — required if journey has multiple segments) | Reservation + booking journey must exist; if multi-segment journey, `flight_number` required; requested seat type must be available in fare class | Reservations (booking segment), Journeys (seat availability + seat type check) | Booking segment `seat` field updated | `seat_assigned`, `preference`, `fare_class`, `flight_number` |
| `add_baggage_allowance` | `confirmation_number` (str, required) <br>`journey_id` (str, required) <br>`num_bags` (int, required, 0–5) <br>`flight_number` (str, optional — if omitted, applies to all segments) | `num_bags` must be 0–5; reservation + booking journey must exist | Reservations | Booking segment(s) `bags_checked` updated | `bags_checked`, `journey_id` |
| `add_meal_request` | `confirmation_number` (str, required) <br>`passenger_id` (str, required) <br>`journey_id` (str, required) <br>`meal_type` (str, required) <br>`flight_number` (str, optional — if omitted, applies to all segments) | `meal_type` must be valid; reservation + booking journey must exist | Reservations | Booking segment(s) `meal_request` updated | `meal_type`, `journey_id` |
| `add_to_standby` | `confirmation_number` (str, required) <br>`journey_id` (str, required) <br>`passenger_ids` (list[str], required) | Journey must exist in journeys table; journey must not be cancelled; all passenger_ids must be valid for the reservation | Reservations, Journeys (by journey_id) | `standby_list` field added/updated on reservation; passengers added to journey's `standby_list` | `journey_id`, `standby_list_position` |
| `issue_travel_credit` | `confirmation_number` (str, required) <br>`passenger_id` (str, required) <br>`amount` (float, required) <br>`credit_reason` (str, required) | Reservation must exist; `credit_reason` must be valid | Reservations (existence check) | New entry added to `travel_credits` table | `credit_code`, `amount`, `valid_months` (12) |
| `issue_hotel_voucher` | `confirmation_number` (str, required) <br>`passenger_id` (str, required) <br>`num_nights` (int, required, max 3) | Reservation must exist; `num_nights` ≤ 3 | Reservations (existence check) | New entry added to `hotel_vouchers` table | `voucher_code`, `number_of_nights`, `valid_at` |
| `issue_meal_voucher` | `confirmation_number` (str, required) <br>`passenger_id` (str, required) <br>`voucher_reason` (str, required) | Reservation must exist; `voucher_reason` must be valid | Reservations (existence check) | New entry added to `meal_vouchers` table | `voucher_code`, `amount` ($12–$25 depending on reason), `valid_at` |
| `process_refund` | `confirmation_number` (str, required) <br>`refund_amount` (float, required, > 0) <br>`refund_type` (str, required) | Reservation must exist; `refund_amount` > 0; `refund_type` must be valid | Reservations (existence check) | New entry added to `refunds` table | `refund_id`, `refund_amount`, `processing_days` (7) |
| `transfer_to_agent` | `confirmation_number` (str, optional) <br>`transfer_reason` (str, required) <br>`issue_summary` (str, required) | If `confirmation_number` provided, reservation must exist | Reservations (existence check, if provided) | None | `transfer_id`, `estimated_wait`, `message` |

---

## Enums Reference

| Field | Valid Values |
|---|---|
| `fare_class` | `basic_economy`, `main_cabin`, `premium_economy`, `business`, `first` |
| `rebooking_type` | `voluntary`, `same_day`, `irrops_cancellation`, `irrops_delay`, `irrops_schedule_change`, `missed_flight_passenger_fault`, `missed_connection_airline_fault` |
| `cancellation_reason` | `voluntary`, `irrops_refund`, `24_hour_rule` |
| `credit_reason` | `cancellation_non_refundable`, `fare_difference_negative`, `service_recovery`, `goodwill`, `downgrade_compensation` |
| `voucher_reason` (meal) | `delay_over_2_hours`, `delay_over_4_hours`, `cancellation_wait_same_day`, `irrops_overnight` |
| `meal_type` | `vegetarian`, `vegan`, `kosher`, `halal`, `gluten_free`, `diabetic`, `low_sodium`, `child`, `hindu`, `none`, `standard` |
| `refund_type` | `full_fare`, `partial_fare`, `taxes_only`, `ancillary_fees` |
| `seat_preference` | `window`, `aisle`, `middle`, `no_preference` |
| `fare_type` | `refundable`, `non_refundable` |
