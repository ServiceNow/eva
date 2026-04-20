## Summary

The ITSM Voice Agent handles inbound calls to an enterprise IT service desk. Employees call to report issues, request hardware and software, manage facilities resources, and handle account and access changes. The agent authenticates callers, retrieves relevant records, walks through troubleshooting when appropriate, attempts direct resolution, submits requests, and completes follow-up actions — all over a voice interface.

| Metric | Value |
| --- | --- |
| Total flows | 21 (+ 1 branching variant for outage) |
| Total tools | 59 |
| Auth tools | 4 |
| Read tools | 23 |
| Write tools | 31 |
| System tools | 1 (transfer to live agent) |
| Avg tool calls per flow | ~4.5 |
| Min tool calls (Flows 9, 11, 12, 13, 20, 21) | 2–3 |
| Max tool calls (Flow 17) | 7 |
| Free-text write params | 0 (all write params are deterministic) |

## Ticket Metadata Conventions

`troubleshooting_completed` on `create_incident_ticket` is a boolean **ticket attribute** (not a flow-level claim). It is `true` for login_issue, network_connectivity, and hardware_malfunction tickets (because a troubleshooting guide is always walked through before ticketing), and `false` for service_outage tickets (which skip guided troubleshooting).

## Authentication Tiers

| Tier | Required For | Tools |
| --- | --- | --- |
| Standard | Incidents, hardware, software, facilities (Flows 1–14, 19–21) | `verify_employee_auth(employee_id, phone_last_four)` |
| Elevated (Standard + OTP) | Group membership, permission change (Flows 16–17) | Standard → `initiate_otp_auth(employee_id)` → `verify_otp_auth(employee_id, otp_code)` |
| Manager + OTP | Account provisioning, access removal (Flows 15, 18) | `verify_manager_auth(employee_id, phone_last_four, manager_auth_code)` → OTP (manager auth tool performs standard verification in the same call) |

## Flow Categories

- **Resolving Issues** (Flows 1–4): Login, outage, hardware malfunction, network/VPN
- **Hardware Requests** (Flows 5–6): Laptop replacement, monitor bundle
- **Software Requests** (Flows 7–10): App access, license (permanent or temporary), renewal, status check
- **Facilities Requests** (Flows 11–14): Desk, parking, ergonomic equipment, conference room
- **Accounts & Access** (Flows 15–18): Provisioning, group membership, permission change, off-boarding
- **Extended Flows** (Flows 19–21): Stolen device, MFA reset, software request status/escalation

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
| 5a (resolved) | `mark_resolved` | `employee_id`, `flow_id` = `login_issue`, `resolution_type` = `resolved_via_troubleshooting` |
| 5b (escalated) | `create_incident_ticket` | `employee_id`, `category` = `login_issue`, `urgency`, `affected_system`, `troubleshooting_completed` = `true` |
| 6b | `assign_sla_tier` | `ticket_number`, `sla_tier` ∈ {`tier_1`, `tier_2`, `tier_3`} |

**Tool calls:** 5 (resolved directly) or 6 (escalated to ticket)

**Note:** Steps 4a/4b are alternatives — agent picks based on what the caller describes. Ask the caller explicitly: *"Is your account locked out (for example, too many failed sign-in attempts), or has your password expired and you can't reset it?"* Choose 4a for a lockout, 4b for a password/reset issue. Steps 5–6 only happen if 4a/4b fail.

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
| 5 | `assign_sla_tier` | `ticket_number`, `sla_tier` ∈ {`tier_1`, `tier_2`, `tier_3`} |
| 6 | `link_known_error` | `ticket_number`, `service_name` |

**Tool calls:** 6

**Note:** The ticket is created *before* `assign_sla_tier` and `link_known_error` because both operations attach metadata to an existing ticket. If the workaround exists, the agent reads it to the caller after the link succeeds.

---

### Flow 3 — Hardware Malfunction

**Premise:** Employee reports a broken device. Agent walks the caller through a short troubleshooting guide (reseat cables, power cycle, try a known-good peripheral). If the issue resolves, close the call without a ticket. Otherwise, look up the asset, create an incident, and schedule field dispatch.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_assets` | `employee_id` |
| 3 | `get_troubleshooting_guide` | `issue_category` = `hardware_malfunction` |
| 4a (resolved) | `mark_resolved` | `employee_id`, `flow_id` = `hardware_malfunction`, `resolution_type` = `resolved_via_troubleshooting` |
| 4b | `get_asset_record` | `asset_tag` (e.g., `AST-LPT-284719`) |
| 5b | `create_incident_ticket` | `employee_id`, `category` = `hardware_malfunction`, `urgency`, `affected_system` = asset tag, `troubleshooting_completed` = `true` |
| 6b | `assign_sla_tier` | `ticket_number`, `sla_tier` ∈ {`tier_1`, `tier_2`, `tier_3`} |
| 7b | `schedule_field_dispatch` | `ticket_number`, `employee_id`, `building_code`, `floor_code`, `preferred_date`, `time_window` ∈ {`morning`, `afternoon`, `full_day`} |

**Tool calls:** 4 (resolved via troubleshooting) or 7 (escalated)

---

### Flow 4 — Network / VPN Issue

**Premise:** Employee can't connect to the network or VPN. Agent walks through troubleshooting, creates a ticket if unresolved, and asks the caller to run a network diagnostic tool and provide the reference code.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_record` | `employee_id` |
| 3 | `get_troubleshooting_guide` | `issue_category` = `network_connectivity` |
| 4a (resolved) | `mark_resolved` | `employee_id`, `flow_id` = `network_connectivity`, `resolution_type` = `resolved_via_troubleshooting` |
| 4b | `create_incident_ticket` | `employee_id`, `category` = `network_connectivity`, `urgency`, `affected_system` ∈ {`vpn`, `wifi`, `ethernet`}, `troubleshooting_completed` = `true` |
| 5b | `assign_sla_tier` | `ticket_number`, `sla_tier` ∈ {`tier_1`, `tier_2`, `tier_3`} |
| 6b | `attach_diagnostic_log` | `ticket_number`, `diagnostic_ref_code` (e.g., `DIAG-4KM29X7B`) |

**Tool calls:** 4 (resolved via troubleshooting) or 6 (ticket created + SLA tier + diagnostic attached)

**Note:** Steps 4b–6b only happen if the troubleshooting guide does not resolve the issue.

---

### Hardware Requests

### Flow 5 — Laptop Replacement

**Premise:** Employee requests a replacement laptop. Agent checks assets, verifies entitlement, verifies the department cost center has budget, submits the request, and generates a return authorization (RMA) for the old device with a 14-day deadline.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_assets` | `employee_id` |
| 3 | `check_hardware_entitlement` | `employee_id`, `request_type` = `laptop_replacement` |
| 4 | `verify_cost_center_budget` | `employee_id` (tool auto-fetches `department_code` and `cost_center_code` from the employee record) |
| 5 | `submit_hardware_request` | `employee_id`, `request_type` = `laptop_replacement`, `justification` ∈ {`end_of_life`, `performance_degradation`, `physical_damage`, `lost_or_stolen`}, `current_asset_tag`, `laptop_os` ∈ {`macos`, `windows`}, `laptop_size` ∈ {`13_inch`, `14_inch`, `16_inch`}, `delivery_building`, `delivery_floor` |
| 6 | `initiate_asset_return` | `employee_id`, `asset_tag`, `request_id` |

**Tool calls:** 6 (step 6 skipped when `justification = lost_or_stolen` — see Flow 19 for that branch)

---

### Flow 6 — Monitor Bundle

**Premise:** Employee requests a monitor setup. Agent checks entitlement, verifies the cost center has budget (auto-fetched from employee record), then submits the request.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_hardware_entitlement` | `employee_id`, `request_type` = `monitor_bundle` |
| 3 | `verify_cost_center_budget` | `employee_id` (tool auto-fetches `department_code` and `cost_center_code` from the employee record) |
| 4 | `submit_hardware_request` | `employee_id`, `request_type` = `monitor_bundle`, `justification` ∈ {`new_setup`, `replacement`}, `monitor_size` ∈ {`24_inch`, `27_inch`, `32_inch`}, `delivery_building`, `delivery_floor` |

**Tool calls:** 4

---

### Software Requests

### Flow 7 — Application Access Request

**Premise:** Employee requests access to a software application. Agent retrieves app details, submits request, and routes the manager approval workflow if required.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_application_details` | `application_name` (e.g., `"Slack Enterprise"` — resolved to `catalog_id` via catalog names + aliases) |
| 3 | `submit_access_request` | `employee_id`, `catalog_id` (from step 2 response), `access_level` ∈ {`read_only`, `standard`, `admin`} |
| 4 | `route_approval_workflow` | `request_id`, `employee_id`, `approver_employee_id` |

**Tool calls:** 3 (no approval required) or 4 (approval required)

---

### Flow 8 — License Request (Permanent)

**Premise:** Employee requests a permanent software license. Agent looks up the license by name, validates the cost center (auto-fetched from employee record), then submits.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_license_catalog_item` | `license_name` (e.g., `"JetBrains IntelliJ IDEA"`) |
| 3 | `validate_cost_center` | `employee_id` (tool auto-fetches CC from record) |
| 4 | `submit_license_request` | `employee_id`, `catalog_id` (from step 2), `duration_days = null` (permanent) |

**Tool calls:** 4

---

### Flow 9 — Temporary License

**Premise:** Employee needs a time-limited license for a project or evaluation. Uses the merged `submit_license_request` tool with `duration_days` set.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_license_catalog_item` | `license_name` |
| 3 | `submit_license_request` | `employee_id`, `catalog_id`, `duration_days` ∈ {`30`, `60`, `90`} |

**Tool calls:** 3

---

### Flow 10 — License Renewal

**Premise:** Employee has an expiring or recently expired license. Must be within 30 days of expiry or ≤14 days past expiry.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_licenses` | `employee_id` |
| 3 | `submit_license_renewal` | `employee_id`, `license_assignment_id` (e.g., `LASGN-048271`) — agent picks the assignment from step 2 matching the product name the caller says; the tool enforces the 30-day / 14-day-expired window |

**Tool calls:** 3

**Note:** The caller communicates the license by **product name** (e.g., "JetBrains IntelliJ IDEA"), not by assignment ID. The agent finds the matching `license_assignment_id` in the response of `get_employee_licenses`.

---

### Facilities Requests

### Flow 11 — Desk / Office Space

**Premise:** Employee needs a desk assignment. Agent checks availability, presents options, assigns chosen desk. If zero desks are available on the requested floor, agent offers to place the caller on the waitlist.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_desk_availability` | `building_code` (e.g., `BLD3`), `floor_code` (e.g., `FL2`) |
| 3a | `submit_desk_assignment` | `employee_id`, `desk_code` (e.g., `BLD3-FL2-D107`) |
| 3b | `submit_waitlist` | `employee_id`, `resource_type = desk`, `zone_or_building` (waitlist branch — only when no desks available) |

**Tool calls:** 3 (3a assignment OR 3b waitlist)

---

### Flow 12 — Parking Space

**Premise:** Employee requests a parking space. Waitlist branch available when zone has no open spots.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_parking_availability` | `zone_code` (e.g., `PZA`) |
| 3a | `submit_parking_assignment` | `employee_id`, `parking_space_id` (e.g., `PZA-042`) |
| 3b | `submit_waitlist` | `employee_id`, `resource_type = parking`, `zone_or_building` (waitlist branch) |

**Tool calls:** 3 (3a OR 3b)

---

### Flow 13 — Ergonomic Equipment

**Premise:** Employee requests ergonomic equipment. For standing desk converters and chairs, a completed ergonomic assessment must be on file first. For keyboards, monitor arms, and footrests, no assessment is required.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_ergonomic_assessment` | `employee_id` (skipped for `ergonomic_keyboard`, `monitor_arm`, `footrest`) |
| 3 | `submit_equipment_request` | `employee_id`, `equipment_type` ∈ {`standing_desk_converter`, `ergonomic_chair`, `ergonomic_keyboard`, `monitor_arm`, `footrest`}, `delivery_building`, `delivery_floor` |

**Tool calls:** 2 (`ergonomic_keyboard`, `monitor_arm`, `footrest`) or 3 (`standing_desk_converter`, `ergonomic_chair`)

---

### Flow 14 — Conference Room Booking

**Premise:** Employee books a conference room. Agent checks availability by building, optional floor, date, time, and capacity, books the room, and creates a calendar event record.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `check_room_availability` | `building_code`, `floor_code` (optional), `date`, `start_time`, `end_time`, `min_capacity` |
| 3 | `submit_room_booking` | `employee_id`, `room_code` (e.g., `BLD3-FL2-RM210`), `date`, `start_time`, `end_time`, `attendee_count` |
| 4 | `send_calendar_invite` | `request_id`, `employee_id`, `room_code`, `date`, `start_time`, `end_time` (persists a `calendar_events` row keyed by event_id) |

**Tool calls:** 4

---

### Accounts & Access

### Flow 15 — New Employee Account Provisioning

**Premise:** A manager calls to set up accounts for a new hire. Manager-tier auth + OTP required. Agent verifies the new hire in HR, confirms no duplicates, provisions access.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_manager_auth` | `employee_id`, `phone_last_four`, `manager_auth_code` (e.g., `K4M2P9`) — performs standard + manager verification in one call |
| 2 | `initiate_otp_auth` | `employee_id` |
| 3 | `verify_otp_auth` | `employee_id`, `otp_code` |
| 4 | `lookup_new_hire` | `new_hire_employee_id` (e.g., `EMP092841`) |
| 5 | `check_existing_accounts` | `employee_id` (new hire) |
| 6 | `provision_new_account` | `manager_employee_id`, `new_hire_employee_id`, `department_code`, `role_code`, `start_date`, `access_groups` (e.g., `["GRP-ENGCORE", "GRP-VPNALL"]`) |

**Tool calls:** 6

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
| 6 | `submit_group_membership_change` | `employee_id`, `group_code`, `action` ∈ {`add`, `remove`} — when approval is required, response includes `request_id` for step 7 |
| 7 | `route_approval_workflow` | `request_id` (from step 6), `employee_id`, `approver_employee_id` — only when the requested group has `requires_approval = true` and `action = add` |

**Tool calls:** 6 (group does not require approval or action is `remove`) or 7 (adding to a group that requires approval)

---

### Flow 17 — Permission Change (Role Change)


**Premise:** Employee's role is changing and permissions need updating. An HR-gated eligibility check confirms the role change has been pre-approved before any permission modification is made. Agent retrieves templates, submits change, schedules the mandatory 90-day access review.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `initiate_otp_auth` | `employee_id` |
| 3 | `verify_otp_auth` | `employee_id`, `otp_code` |
| 4 | `check_role_change_authorized` | `employee_id`, `new_role_code` — must return success before proceeding |
| 5 | `get_permission_templates` | `role_code` (e.g., `SWE`) |
| 6 | `submit_permission_change` | `employee_id`, `new_role_code`, `permission_template_id` (e.g., `PTPL-SWE-02`), `effective_date` (must match HR-approved date from step 4) |
| 7 | `schedule_access_review` | `case_id`, `employee_id`, `review_date` (effective_date + 90 days, ±3 days) |

**Tool calls:** 7

---

### Flow 18 — Access Removal (Off-boarding)

**Premise:** Manager removes all access for a departing employee. Manager-tier auth + OTP required. Agent confirms HR off-boarding record, removes access, initiates hardware recovery.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_manager_auth` | `employee_id`, `phone_last_four`, `manager_auth_code` — performs standard + manager verification in one call |
| 2 | `initiate_otp_auth` | `employee_id` |
| 3 | `verify_otp_auth` | `employee_id`, `otp_code` |
| 4 | `get_offboarding_record` | `employee_id` (departing, e.g., `EMP072948`) |
| 5 | `submit_access_removal` | `manager_employee_id`, `departing_employee_id`, `last_working_day` (must not be in the past), `removal_scope` ∈ {`full`, `staged`} |
| 6 | `initiate_asset_recovery` | `departing_employee_id`, `case_id`, `recovery_method` ∈ {`shipping_label`, `drop_off`} |

**Tool calls:** 6

---

### Extended Flows

### Flow 19 — Security Incident / Stolen Device

**Premise:** Employee reports a company device as lost, stolen, or potentially compromised. Agent files a security incident, triggers a remote wipe, and submits a replacement hardware request flagged as `lost_or_stolen` (no asset return since the device isn't recoverable).

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_employee_assets` | `employee_id` |
| 3 | `report_security_incident` | `employee_id`, `asset_tag`, `incident_type` ∈ {`lost`, `stolen`, `suspected_compromise`} |
| 4 | `initiate_remote_wipe` | `asset_tag`, `security_case_id` |
| 5 | `submit_hardware_request` | `employee_id`, `request_type = laptop_replacement`, `justification = lost_or_stolen`, `current_asset_tag`, `laptop_os`, `laptop_size`, `delivery_building`, `delivery_floor` |

**Tool calls:** 5

**Note:** Unlike Flow 5, there is no `initiate_asset_return` — the asset is not recoverable.

---

### Flow 20 — MFA Reset / Lost Device (Refusal + Escalation)

**Premise:** Employee replaced their phone; OTP cannot be sent to the old number. Policy requires in-person identity verification for phone-of-record changes. Agent confirms the scenario, explains the in-person-only policy, and offers a transfer to a live agent. No DB mutation happens beyond the security case record.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` (succeeds against *current* number on file) |
| 2 | `submit_mfa_reset` | `employee_id`, `new_phone_last_four` — returns `in_person_required` error with guidance |
| 3 | `transfer_to_agent` | `employee_id`, `transfer_reason = policy_exception_needed`, `issue_summary` |

**Tool calls:** 3

**Note:** The agent's correct behavior is to receive the `in_person_required` error and explain it, not to retry. The transfer tool is the terminating step.

---

### Flow 21 — Software Request Status Check / Approval Escalation

**Premise:** Employee previously submitted a software access request; manager hasn't approved within the 48-hour SLA. Employee calls to check status and requests escalation to the skip-level manager.

| Step | Tool | Key Params |
| --- | --- | --- |
| 1 | `verify_employee_auth` | `employee_id`, `phone_last_four` |
| 2 | `get_request_status` | `request_id` (e.g., `REQ-SW-048271`) |
| 3 | `escalate_approval` | `request_id`, `escalate_to_employee_id` (skip-level manager) — only when the SLA has been exceeded |

**Tool calls:** 2 (within SLA, status-only) or 3 (SLA exceeded, escalation)

---

## Tool Call Distribution

| Tool Calls | Flows |
| --- | --- |
| 2 | Flow 13 (Equipment — no-assessment branch), Flow 21 (Status check — within SLA) |
| 3 | Flow 7 (App Access — no approval), Flow 9 (Temp License), Flow 10 (Renewal), Flow 11 (Desk), Flow 12 (Parking), Flow 13 (Equipment — assessment branch), Flow 20 (MFA Reset), Flow 21 (Escalation) |
| 4 | Flow 2a (Outage existing), Flow 3 (HW Malfunction — resolved), Flow 4 (Network — resolved), Flow 6 (Monitor), Flow 7 (App Access — with approval), Flow 8 (License), Flow 14 (Room) |
| 5 | Flow 1 (Login — resolved), Flow 19 (Security Incident) |
| 6 | Flow 1 (Login — escalated), Flow 2b (Outage new), Flow 4 (Network — ticketed), Flow 5 (Laptop), Flow 15 (Provisioning), Flow 16 (Group — no approval or remove), Flow 18 (Access Removal) |
| 7 | Flow 3 (HW Malfunction — escalated), Flow 16 (Group — approval required), Flow 17 (Permission Change) |

Note: Flow 5 (Laptop Replacement) now includes `verify_cost_center_budget` (6 steps; 5 when justification=lost_or_stolen, since the asset return step is dropped — see Flow 19). Flow 10 (License Renewal) collapsed to 3 steps after merging `check_renewal_eligibility` into `submit_license_renewal`. Flow 1 resolved, Flow 3 resolved, and Flow 4 resolved each add a terminal `mark_resolved` step to make resolution state-observable (interactions table) instead of implicit. Flows 1b, 2b, 3b, and 4b all include `assign_sla_tier` after `create_incident_ticket` per the unified incident-ticket SLA policy.

## Key Entity Types Communicated by Caller

These are the identifiers the user must communicate over voice — the core challenge for voice agent evaluation:

| Entity | Format | Example | Flows |
| --- | --- | --- | --- |
| Employee ID | `EMP` + 6 digits | `EMP048271` | All |
| Phone last four | 4 digits | `7294` | All |
| OTP code | 6 digits | `839201` | 15–18 |
| Manager auth code | 6 alphanumeric uppercase | `K4M2P9` | 15, 18 |
| Asset tag | `AST-XXX-NNNNNN` | `AST-LPT-284719` | 3, 5, 19 |
| Application name | free-form (exact-matched against catalog names + aliases) | `"Slack Enterprise"` | 7 |
| License name | free-form (exact-matched against catalog names + aliases) | `"JetBrains IntelliJ IDEA"` | 8, 9 |
| License assignment ID | `LASGN-NNNNNN` | `LASGN-048271` | 10 |
| Request ID | `REQ-XX-NNNNNN` | `REQ-SW-048271` | 21 |
| Building | code `BLDNN` or name/alias (e.g. `"Downtown"`, `"HQ"`, `"East Campus"`) | `BLD3` or `"Headquarters"` | 3, 5, 6, 11, 13, 14, 19 |
| Floor code | `FLNN` | `FL2` | 3, 5, 6, 11, 13, 14, 19 |
| Parking zone | code `PZX` or name/alias (e.g. `"Executive Garage"`, `"North Lot"`) | `PZA` or `"Executive Garage"` | 12 |
| Group code | `GRP-XXXXXX` | `GRP-DBREAD` | 15, 16 |
| Permission template ID | `PTPL-XXX-NN` | `PTPL-SWE-02` | 17 |
| Diagnostic ref code | `DIAG-XXXXXXXX` | `DIAG-4KM29X7B` | 4 |
| Target system | enum | `active_directory` | 1 |
| Service name | enum | `email_exchange` | 2 |
| Dates | `YYYY-MM-DD` | `2026-08-15` | 3, 5, 14, 15, 17, 18 |
| Times | `HH:MM` | `09:00` | 14 |
