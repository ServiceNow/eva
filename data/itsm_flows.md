
## Summary

The ITSM Voice Agent handles inbound calls to an enterprise IT service desk. Employees call to report issues, request hardware and software, manage facilities resources, and handle account and access changes. The agent authenticates callers, retrieves relevant records, walks through troubleshooting when appropriate, attempts direct resolution, submits requests, and completes follow-up actions — all over a voice interface.

| Metric | Value |
| --- | --- |
| Total flows | 18 (+ 1 branching variant for outage) |
| Total tools | 53 |
| Auth tools | 4 |
| Read tools | 21 |
| Write tools | 27 |
| System tools | 1 (transfer to live agent) |
| Avg tool calls per flow | 4.5 |
| Min tool calls (Flows 9, 11, 12, 13) | 3 |
| Max tool calls (Flow 15) | 7 |
| Free-text write params | 0 (all write params are deterministic) |

## Authentication Tiers

| Tier | Required For | Tools |
| --- | --- | --- |
| Standard | Incidents, hardware, software, facilities (Flows 1–14) | `verify_employee_auth(employee_id, phone_last_four)` |
| Elevated (Standard + OTP) | Group membership, permission change (Flows 16–17) | Standard → `initiate_otp_auth(employee_id)` → `verify_otp_auth(employee_id, otp_code)` |
| Manager + OTP | Account provisioning, access removal (Flows 15, 18) | Standard → `verify_manager_auth(employee_id, manager_auth_code)` → OTP |

## Flow Categories

- **Resolving Issues** (Flows 1–4): Login, outage, hardware malfunction, network/VPN
- **Hardware Requests** (Flows 5–6): Laptop replacement, monitor bundle
- **Software Requests** (Flows 7–10): App access, license, temporary license, renewal
- **Facilities Requests** (Flows 11–14): Desk, parking, ergonomic equipment, conference room
- **Accounts & Access** (Flows 15–18): Provisioning, group membership, permission change, off-boarding

---

## Flow Details

### Resolving Issues

### Flow 1 — Login Issue

**Premise:** Employee cannot log into a system. Agent authenticates, retrieves troubleshooting steps, walks caller through each step, then attempts direct resolution (account unlock or password reset). Only creates a ticket + SLA assignment if direct resolution fails.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_record` | `employee_id` |
| 3 | `get_troubleshooting_guide` | `issue_category` = `login_issue` |
| 4a | `attempt_account_unlock` | `employee_id`, `target_system` ∈ {`active_directory`, `sso_identity`, `email_exchange`, `vpn_gateway`, `erp_oracle`} |
| 4b | `attempt_password_reset` | `employee_id`, `target_system` (same enum) |
| 5 | `create_incident_ticket` | `employee_id`, `category` = `login_issue`, `urgency`, `affected_system`, `troubleshooting_completed` = `true` |
| 6 | `assign_sla_tier` | `ticket_number`, `sla_tier` ∈ {`tier_1`, `tier_2`, `tier_3`} |

**Tool calls:** 5 (resolved directly) or 7 (escalated to ticket)

**Note:** Steps 4a/4b are alternatives — agent picks based on the issue. Steps 5–6 only happen if 4a/4b fail.

---

### Flow 2a — Service Outage (existing outage found)

**Premise:** Employee reports a service is down. Agent checks for an existing outage and adds the caller to the affected users list instead of creating a duplicate ticket.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_record` | `employee_id` |
| 3 | `check_existing_outage` | `service_name` ∈ {`email_exchange`, `vpn_gateway`, `erp_oracle`, ...} |
| 4 | `add_affected_user` | `ticket_number`, `employee_id` |

**Tool calls:** 4

---

### Flow 2b — Service Outage (no existing outage)

**Premise:** Employee reports a service outage not yet logged. Agent creates a new incident and checks the known error database for a matching workaround.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_record` | `employee_id` |
| 3 | `check_existing_outage` | `service_name` |
| 4 | `create_incident_ticket` | `employee_id`, `category` = `service_outage`, `urgency`, `affected_system`, `troubleshooting_completed` = `false` |
| 5 | `link_known_error` | `ticket_number`, `service_name` |

**Tool calls:** 5

---

### Flow 3 — Hardware Malfunction

**Premise:** Employee reports a broken device. Agent looks up assigned assets, confirms the device, creates an incident, and schedules a field technician dispatch for on-site inspection.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_assets` | `employee_id` |
| 3 | `get_asset_record` | `asset_tag` (e.g., `AST-LPT-284719`) |
| 4 | `create_incident_ticket` | `employee_id`, `category` = `hardware_malfunction`, `urgency`, `affected_system` = asset tag, `troubleshooting_completed` = `false` |
| 5 | `schedule_field_dispatch` | `ticket_number`, `employee_id`, `building_code`, `floor_code`, `preferred_date`, `time_window` ∈ {`morning`, `afternoon`, `full_day`} |

**Tool calls:** 5

---

### Flow 4 — Network / VPN Issue

**Premise:** Employee can't connect to the network or VPN. Agent walks through troubleshooting, creates a ticket if unresolved, and asks the caller to run a network diagnostic tool and provide the reference code.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_record` | `employee_id` |
| 3 | `get_troubleshooting_guide` | `issue_category` = `network_connectivity` |
| 4 | `create_incident_ticket` | `employee_id`, `category` = `network_connectivity`, `urgency`, `affected_system` ∈ {`vpn`, `wifi`, `ethernet`}, `troubleshooting_completed` = `true` |
| 5 | `attach_diagnostic_log` | `ticket_number`, `diagnostic_ref_code` (e.g., `DIAG-4KM29X7B`) |

**Tool calls:** 5

---

### Hardware Requests

### Flow 5 — Laptop Replacement

**Premise:** Employee requests a replacement laptop. Agent checks assets, verifies entitlement, submits request, and generates a return authorization (RMA) for the old device with a 14-day deadline.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_assets` | `employee_id` |
| 3 | `check_hardware_entitlement` | `employee_id`, `request_type` = `laptop_replacement` |
| 4 | `submit_hardware_request` | `employee_id`, `request_type` = `laptop_replacement`, `replacement_reason` ∈ {`end_of_life`, `performance_degradation`, `physical_damage`}, `current_asset_tag`, `delivery_building`, `delivery_floor` |
| 5 | `initiate_asset_return` | `employee_id`, `asset_tag`, `request_id` |

**Tool calls:** 5

---

### Flow 6 — Monitor Bundle

**Premise:** Employee requests a monitor setup. Agent checks entitlement, verifies the cost center has budget (caller provides CC code), then submits the request.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_hardware_entitlement` | `employee_id`, `request_type` = `monitor_bundle` |
| 3 | `verify_cost_center_budget` | `department_code`, `cost_center_code` (e.g., `CC-4022`) |
| 4 | `submit_hardware_request` | `employee_id`, `request_type` = `monitor_bundle`, `replacement_reason` ∈ {`new_setup`, `replacement`}, `monitor_size` ∈ {`24_inch`, `27_inch`, `32_inch`}, `delivery_building`, `delivery_floor` |

**Tool calls:** 4

---

### Software Requests

### Flow 7 — Application Access Request

**Premise:** Employee requests access to a software application. Agent retrieves app details, submits request, and routes the manager approval workflow if required.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_application_details` | `catalog_id` (e.g., `APP-0042`) |
| 3 | `submit_access_request` | `employee_id`, `catalog_id`, `access_level` ∈ {`read_only`, `standard`, `admin`} |
| 4 | `route_approval_workflow` | `request_id`, `employee_id`, `approver_employee_id` |

**Tool calls:** 4 (step 4 only when approval required)

---

### Flow 8 — License Request

**Premise:** Employee requests a permanent software license. Agent looks up the license, validates the cost center (caller provides CC code), then submits.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_license_catalog_item` | `catalog_id` (e.g., `LIC-0018`) |
| 3 | `validate_cost_center` | `cost_center_code` (e.g., `CC-4021`), `department_code` |
| 4 | `submit_license_request` | `employee_id`, `catalog_id` |

**Tool calls:** 4

---

### Flow 9 — Temporary License

**Premise:** Employee needs a time-limited license for a project or evaluation.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_license_catalog_item` | `catalog_id` |
| 3 | `submit_temporary_license` | `employee_id`, `catalog_id`, `duration_days` ∈ {`30`, `60`, `90`} |

**Tool calls:** 3

---

### Flow 10 — License Renewal

**Premise:** Employee has an expiring or recently expired license. Must be within 30 days of expiry or ≤14 days past.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_licenses` | `employee_id` |
| 3 | `check_renewal_eligibility` | `employee_id`, `license_assignment_id` (e.g., `LASGN-048271`) |
| 4 | `submit_license_renewal` | `employee_id`, `license_assignment_id` |

**Tool calls:** 4

---

### Facilities Requests

### Flow 11 — Desk / Office Space

**Premise:** Employee needs a desk assignment. Agent checks availability, presents options, assigns chosen desk.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_space_availability` | `building_code` (e.g., `BLD3`), `floor_code` (e.g., `FL2`) |
| 3 | `submit_desk_assignment` | `employee_id`, `desk_code` (e.g., `BLD3-FL2-D107`) |

**Tool calls:** 3

---

### Flow 12 — Parking Space

**Premise:** Employee requests a parking space.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_parking_availability` | `zone_code` (e.g., `PZA`) |
| 3 | `submit_parking_assignment` | `employee_id`, `parking_space_id` (e.g., `PZA-042`) |

**Tool calls:** 3

---

### Flow 13 — Ergonomic Equipment

**Premise:** Employee requests ergonomic equipment. For standing desk converters and chairs, a completed ergonomic assessment must be on file first.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_ergonomic_assessment` | `employee_id` |
| 3 | `submit_equipment_request` | `employee_id`, `equipment_type` ∈ {`standing_desk_converter`, `ergonomic_chair`, `ergonomic_keyboard`, `monitor_arm`, `footrest`}, `delivery_building`, `delivery_floor` |

**Tool calls:** 3

---

### Flow 14 — Conference Room Booking

**Premise:** Employee books a conference room. Agent checks availability by building, floor, date, time, and capacity, books the room, and sends a calendar invite.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_room_availability` | `building_code`, `floor_code`, `date`, `start_time`, `end_time`, `min_capacity` |
| 3 | `submit_room_booking` | `employee_id`, `room_code` (e.g., `BLD3-FL2-RM210`), `date`, `start_time`, `end_time`, `attendee_count` |
| 4 | `send_calendar_invite` | `request_id`, `employee_id`, `room_code`, `date`, `start_time`, `end_time` |

**Tool calls:** 4

---

### Accounts & Access

### Flow 15 — New Employee Account Provisioning

**Premise:** A manager calls to set up accounts for a new hire. Full three-tier auth required. Agent verifies the new hire in HR, confirms no duplicates, provisions access.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `verify_manager_auth` | `employee_id`, `manager_auth_code` (e.g., `K4M2P9`) |
| 3 | `initiate_otp_auth` | `employee_id` |
| 4 | `verify_otp_auth` | `employee_id`, `otp_code` |
| 5 | `lookup_new_hire` | `new_hire_employee_id` (e.g., `EMP092841`) |
| 6 | `check_existing_accounts` | `employee_id` (new hire) |
| 7 | `provision_new_account` | `manager_employee_id`, `new_hire_employee_id`, `department_code`, `role_code`, `start_date`, `access_groups` (e.g., `["GRP-ENGCORE", "GRP-VPNALL"]`) |

**Tool calls:** 7

---

### Flow 16 — Group Membership Request

**Premise:** Employee requests to join or leave a system access group. Elevated auth required.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `initiate_otp_auth` | `employee_id` |
| 3 | `verify_otp_auth` | `employee_id`, `otp_code` |
| 4 | `get_group_memberships` | `employee_id` |
| 5 | `get_group_details` | `group_code` (e.g., `GRP-DBREAD`) |
| 6 | `submit_group_membership_change` | `employee_id`, `group_code`, `action` ∈ {`add`, `remove`} |

**Tool calls:** 6

---

### Flow 17 — Permission Change (Role Change)

**Premise:** Employee's role is changing and permissions need updating. Agent retrieves templates, submits change, schedules a mandatory 90-day access review.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `initiate_otp_auth` | `employee_id` |
| 3 | `verify_otp_auth` | `employee_id`, `otp_code` |
| 4 | `get_permission_templates` | `role_code` (e.g., `SWE`) |
| 5 | `submit_permission_change` | `employee_id`, `new_role_code`, `permission_template_id` (e.g., `PTPL-SWE-02`), `effective_date` |
| 6 | `schedule_access_review` | `case_id`, `employee_id`, `review_date` (~90 days after effective date) |

**Tool calls:** 6

---

### Flow 18 — Access Removal (Off-boarding)

**Premise:** Manager removes all access for a departing employee. Full three-tier auth. Agent confirms HR off-boarding record, removes access, initiates hardware recovery.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `verify_manager_auth` | `employee_id`, `manager_auth_code` |
| 3 | `initiate_otp_auth` | `employee_id` |
| 4 | `verify_otp_auth` | `employee_id`, `otp_code` |
| 5 | `get_offboarding_record` | `employee_id` (departing, e.g., `EMP072948`) |
| 6 | `submit_access_removal` | `manager_employee_id`, `departing_employee_id`, `last_working_day`, `removal_scope` ∈ {`full`, `staged`} |
| 7 | `initiate_asset_recovery` | `departing_employee_id`, `case_id`, `recovery_method` ∈ {`office_pickup`, `shipping_label`, `drop_off`} |

**Tool calls:** 7

---

## Tool Call Distribution

| Tool Calls | Flows |
| --- | --- |
| 3 | Flow 9 (Temp License), Flow 11 (Desk), Flow 12 (Parking), Flow 13 (Equipment) |
| 4 | Flow 2a (Outage existing), Flow 6 (Monitor), Flow 7 (App Access), Flow 8 (License), Flow 10 (Renewal), Flow 14 (Room) |
| 5 | Flow 1 (Login — resolved), Flow 2b (Outage new), Flow 3 (HW Malfunction), Flow 4 (Network), Flow 5 (Laptop) |
| 6 | Flow 16 (Group Membership), Flow 17 (Permission Change) |
| 7 | Flow 1 (Login — escalated), Flow 15 (Provisioning), Flow 18 (Access Removal) |

## Key Entity Types Communicated by Caller

These are the identifiers the user must communicate over voice — the core challenge for voice agent evaluation:

| Entity | Format | Example | Flows |
| --- | --- | --- | --- |
| Employee ID | `EMP` + 6 digits | `EMP048271` | All |
| Phone last four | 4 digits | `7294` | All |
| OTP code | 6 digits | `839201` | 15–18 |
| Manager auth code | 6 alphanumeric uppercase | `K4M2P9` | 15, 18 |
| Asset tag | `AST-XXX-NNNNNN` | `AST-LPT-284719` | 3, 5 |
| App catalog ID | `APP-NNNN` | `APP-0042` | 7 |
| License catalog ID | `LIC-NNNN` | `LIC-0018` | 8, 9 |
| License assignment ID | `LASGN-NNNNNN` | `LASGN-048271` | 10 |
| Cost center code | `CC-NNNN` | `CC-4021` | 6, 8 |
| Building code | `BLDNN` | `BLD3` | 3, 5, 6, 11, 13, 14 |
| Floor code | `FLNN` | `FL2` | 3, 5, 6, 11, 13, 14 |
| Parking zone | `PZX` | `PZA` | 12 |
| Group code | `GRP-XXXXXX` | `GRP-DBREAD` | 15, 16 |
| Permission template ID | `PTPL-XXX-NN` | `PTPL-SWE-02` | 17 |
| Diagnostic ref code | `DIAG-XXXXXXXX` | `DIAG-4KM29X7B` | 4 |
| Target system | enum | `active_directory` | 1 |
| Service name | enum | `email_exchange` | 2 |
| Dates | `YYYY-MM-DD` | `2026-08-15` | 3, 5, 14, 15, 17, 18 |
| Times | `HH:MM` | `09:00` | 14 |