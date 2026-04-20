#!/usr/bin/env python3
"""Streamlit app for ITSM domain review.

Reviewers input their name, review flows/policies/tools/scenario data,
and submit structured feedback saved as JSON.

Usage:
    streamlit run apps/itsm_review.py
"""

import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="ITSM Domain Review",
    page_icon=":wrench:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _set_sidebar_width(width_px: int) -> None:
    """Inject CSS to force sidebar to a specific width."""
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            min-width: {width_px}px !important;
            max-width: {width_px}px !important;
            width: {width_px}px !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            width: {width_px}px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Tool descriptions (human-readable pseudocode for each tool)
# ============================================================================

TOOL_DESCRIPTIONS: dict[str, list[str]] = {
    # --- Auth ---
    "verify_employee_auth": [
        "Looks up employee by employee_id in the employees table",
        "Checks if phone_last_four matches the stored value",
        "On success: sets session.employee_auth = True, stores authenticated_employee_id",
        "Returns employee first/last name and ID",
    ],
    "initiate_otp_auth": [
        "Requires session.employee_auth = True (standard auth must pass first)",
        "Looks up employee to retrieve phone_last_four",
        "Sets session.otp_issued = True",
        "Returns confirmation that OTP was sent to phone ending in last 4 digits",
    ],
    "verify_otp_auth": [
        "Checks that initiate_otp_auth was called first (session.otp_issued)",
        "Looks up employee and compares provided otp_code to stored value",
        "On success: sets session.otp_auth = True",
        "Returns authentication confirmation",
    ],
    "verify_manager_auth": [
        "Requires session.employee_auth = True",
        "Looks up employee and compares manager_auth_code to stored value",
        "Validates the employee has is_manager = True",
        "On success: sets session.manager_auth = True, stores manager_employee_id",
        "Returns department_code and list of direct_reports",
    ],
    # --- Shared Lookups ---
    "get_employee_record": [
        "Requires session.employee_auth = True",
        "Looks up employee by employee_id",
        "Returns a filtered subset of fields: employee_id, name, department, role, hire date, status, location, manager",
    ],
    "get_asset_record": [
        "Requires session.employee_auth = True",
        "Looks up asset by asset_tag in the assets table",
        "Returns full asset record (deep copy)",
    ],
    "get_employee_assets": [
        "Requires session.employee_auth = True",
        "Scans all assets where assigned_employee_id matches the given employee_id",
        "Returns list of all matching asset records",
    ],
    # --- Flow 1: Login Issue ---
    "get_troubleshooting_guide": [
        "Requires session.employee_auth = True",
        "Looks up troubleshooting_guides by issue_category (login_issue or network_connectivity)",
        "Returns the ordered list of troubleshooting steps",
    ],
    "attempt_account_unlock": [
        "Requires session.employee_auth = True",
        "Looks up employee's account_status for the specified target_system",
        "Checks if account is locked (locked = True)",
        "If locked: sets locked = False in the DB and returns resolved = True",
        "If not locked: returns error 'not_locked'",
    ],
    "attempt_password_reset": [
        "Requires session.employee_auth = True",
        "Looks up employee's account_status for the specified target_system",
        "Checks if requires_in_person_reset is set (returns error if so)",
        "Sets password_expired = False, temp_password_issued = True",
        "Returns success with message about temporary password",
    ],
    # --- Incidents (Flows 1-4) ---
    "create_incident_ticket": [
        "Requires session.employee_auth = True",
        "For login_issue or network_connectivity: requires troubleshooting_completed = True",
        "Increments _ticket_counter, generates ticket number INC + 7 digits",
        "Creates a ticket record in tickets table with all params + status 'open'",
        "Returns the generated ticket_number",
    ],
    "assign_sla_tier": [
        "Requires session.employee_auth = True",
        "Looks up ticket by ticket_number",
        "Maps SLA tier to response/resolution hours: tier_1 (1h/4h), tier_2 (4h/8h), tier_3 (8h/24h)",
        "Updates the ticket with sla_tier, sla_response_hours, sla_resolution_hours",
    ],
    # --- Flow 2: Outage ---
    "check_existing_outage": [
        "Requires session.employee_auth = True",
        "Looks up active_outages by service_name",
        "If outage exists: returns existing_outage = True, ticket_number, affected_user_count",
        "If no outage: returns existing_outage = False",
    ],
    "add_affected_user": [
        "Requires session.employee_auth = True",
        "Finds active outage matching the ticket_number",
        "Appends employee_id to affected_users list (avoids duplicates)",
        "Updates affected_user_count on the outage",
    ],
    "link_known_error": [
        "Requires session.employee_auth = True",
        "Looks up ticket by ticket_number",
        "Looks up known_errors by service_name",
        "If found: sets ticket.linked_known_error_id, returns workaround text",
        "If no known error: returns linked = False",
    ],
    # --- Flow 3: Hardware Malfunction ---
    "schedule_field_dispatch": [
        "Requires session.employee_auth = True",
        "Looks up ticket by ticket_number",
        "Checks field_dispatch_availability for preferred_date + time_window",
        "If not available: returns up to 3 alternative slots",
        "If available: marks slot as unavailable, generates dispatch ID, updates ticket",
    ],
    # --- Flow 4: Network ---
    "attach_diagnostic_log": [
        "Requires session.employee_auth = True",
        "Looks up ticket by ticket_number",
        "Attaches diagnostic_ref_code to the ticket, sets diagnostic_attached = True",
    ],
    # --- Flows 5-6: Hardware Requests ---
    "check_hardware_entitlement": [
        "Requires session.employee_auth = True",
        "Looks up employee's hardware_entitlements for the given request_type",
        "Checks for existing pending_request (returns error if one exists)",
        "Returns eligibility status, current_asset_tag, and device_age_months",
    ],
    "submit_hardware_request": [
        "Requires session.employee_auth = True",
        "For laptop_replacement: validates replacement_reason is one of (end_of_life, performance_degradation, physical_damage) and current_asset_tag is provided",
        "For monitor_bundle: validates replacement_reason is (new_setup or replacement) and monitor_size is provided",
        "Generates request ID (REQ-HW-NNNNNN), creates request record",
        "Marks pending_request = True on the employee's hardware entitlement",
    ],
    "initiate_asset_return": [
        "Requires session.employee_auth = True",
        "Looks up asset by asset_tag",
        "Calculates return deadline as current_date + 14 days",
        "Generates RMA ID, attaches return_authorization to the asset record",
    ],
    "verify_cost_center_budget": [
        "Requires session.employee_auth = True",
        "Looks up cost center by cost_center_code",
        "Validates department_code matches the cost center's department",
        "Checks has_budget flag, returns remaining_budget_usd",
    ],
    # --- Flow 7: App Access ---
    "get_application_details": [
        "Requires session.employee_auth = True",
        "Looks up software_catalog.applications by catalog_id",
        "Returns full application metadata (deep copy)",
    ],
    "submit_access_request": [
        "Requires session.employee_auth = True",
        "Looks up application by catalog_id, validates access_level is in available_access_levels",
        "Checks requires_manager_approval on the application",
        "Generates request ID (REQ-SW-NNNNNN), creates request with status pending_approval or approved",
    ],
    "route_approval_workflow": [
        "Requires session.employee_auth = True",
        "Looks up request by request_id",
        "Validates approver_employee_id exists in employees",
        "Sets approval_routed_to and 48-hour approval_sla_deadline on the request",
    ],
    # --- Flows 8-9: License ---
    "get_license_catalog_item": [
        "Requires session.employee_auth = True",
        "Looks up software_catalog.licenses by catalog_id",
        "Returns full license catalog metadata (deep copy)",
    ],
    "validate_cost_center": [
        "Requires session.employee_auth = True",
        "Looks up cost center by cost_center_code",
        "Validates department_code matches the cost center's department",
        "Returns validation confirmation (does not check budget amount)",
    ],
    "submit_license_request": [
        "Requires session.employee_auth = True",
        "Generates request ID (REQ-SW-NNNNNN)",
        "Creates request record with request_type = 'license_request' and status 'submitted'",
    ],
    "submit_temporary_license": [
        "Requires session.employee_auth = True",
        "Calculates expiration_date as current_date + duration_days (30, 60, or 90)",
        "Generates request ID, creates request with request_type = 'temporary_license'",
        "Returns expiration_date in the response",
    ],
    # --- Flow 10: Renewal ---
    "get_employee_licenses": [
        "Requires session.employee_auth = True",
        "Looks up employee's software_licenses array",
        "Returns list of all license assignments (deep copy)",
    ],
    "check_renewal_eligibility": [
        "Requires session.employee_auth = True",
        "Finds the license assignment matching license_assignment_id",
        "Calculates days until expiration from _current_date",
        "Eligible if within 30 days before expiry or up to 14 days past expiry",
        "Returns error if too early (>30 days) or too late (>14 days past)",
    ],
    "submit_license_renewal": [
        "Requires session.employee_auth = True",
        "Finds the license assignment by license_assignment_id",
        "Sets new expiration_date = max(current_expiry, today) + 365 days",
        "Updates license status to 'active'",
    ],
    # --- Flow 11: Desk ---
    "check_space_availability": [
        "Requires session.employee_auth = True",
        "Filters facilities.desks by building_code, floor_code, and status = 'available'",
        "Returns list of available desks with desk_code, zone, and near_window flag",
    ],
    "submit_desk_assignment": [
        "Requires session.employee_auth = True",
        "Looks up desk by desk_code, checks status = 'available'",
        "Sets desk status to 'assigned' and records assigned_employee_id",
        "Returns generated request ID (REQ-FAC-NNNNNN)",
    ],
    # --- Flow 12: Parking ---
    "check_parking_availability": [
        "Requires session.employee_auth = True",
        "Filters facilities.parking by zone_code and status = 'available'",
        "Returns list of available spaces with parking_space_id, level, and covered flag",
    ],
    "submit_parking_assignment": [
        "Requires session.employee_auth = True",
        "Looks up parking space by parking_space_id, checks status = 'available'",
        "Sets space status to 'assigned' and records assigned_employee_id",
        "Returns generated request ID (REQ-FAC-NNNNNN)",
    ],
    # --- Flow 13: Ergonomic Equipment ---
    "check_ergonomic_assessment": [
        "Requires session.employee_auth = True",
        "Looks up employee's ergonomic_assessment field",
        "Checks that assessment exists and status = 'completed'",
        "Returns assessment_date if on file; error if not completed",
    ],
    "submit_equipment_request": [
        "Requires session.employee_auth = True",
        "Generates request ID (REQ-FAC-NNNNNN)",
        "Creates request record with equipment_type and delivery location",
    ],
    # --- Flow 14: Conference Room ---
    "check_room_availability": [
        "Requires session.employee_auth = True",
        "Filters facilities.conference_rooms by building, floor, and min_capacity",
        "For each room, checks existing bookings for time overlap on the requested date",
        "Returns list of available rooms with capacity and equipment info",
    ],
    "submit_room_booking": [
        "Requires session.employee_auth = True",
        "Looks up room by room_code, checks for booking conflicts on date/time",
        "Appends new booking to the room's bookings list",
        "Returns generated request ID (REQ-FAC-NNNNNN)",
    ],
    "send_calendar_invite": [
        "Requires session.employee_auth = True",
        "Generates calendar event ID from the request_id",
        "Returns calendar_event_id with room and date/time details",
        "Note: does not modify the database beyond returning the event ID",
    ],
    # --- Flow 15: Provisioning ---
    "lookup_new_hire": [
        "Requires session.manager_auth = True",
        "Looks up employee by new_hire_employee_id",
        "Validates employment_status = 'pending_start'",
        "Returns basic employee info (name, department, role, start date)",
    ],
    "check_existing_accounts": [
        "Requires session.manager_auth = True",
        "Looks up employee's system_accounts",
        "Checks if any accounts have status = 'active'",
        "Returns error if active accounts already exist; success if ready to provision",
    ],
    "provision_new_account": [
        "Requires session.manager_auth = True AND session.otp_auth = True",
        "Validates new hire is in the manager's direct_reports list",
        "Creates system_accounts entry with provisioned access_groups",
        "Sets employment_status to 'active'",
        "Generates company email from first.last@company.com",
        "Returns case ID (CASE-ACCT-NNNNNN)",
    ],
    # --- Flow 16: Group Membership ---
    "get_group_memberships": [
        "Requires session.otp_auth = True",
        "Looks up employee's group_memberships array",
        "Returns list of all group memberships (deep copy)",
    ],
    "get_group_details": [
        "Requires session.otp_auth = True",
        "Looks up access_groups by group_code",
        "Returns full group metadata (deep copy)",
    ],
    "submit_group_membership_change": [
        "Requires session.otp_auth = True",
        "For action = 'add': checks employee is not already a member, checks requires_approval on the group",
        "If no approval needed: appends new membership to employee's group_memberships",
        "For action = 'remove': removes matching entry from group_memberships array",
        "Returns case ID (CASE-ACCT-NNNNNN)",
    ],
    # --- Flow 17: Permission Change ---
    "get_permission_templates": [
        "Requires session.otp_auth = True",
        "Filters permission_templates where role_code matches",
        "Returns all matching templates (deep copy)",
    ],
    "submit_permission_change": [
        "Requires session.otp_auth = True",
        "Validates permission_template_id exists and its role_code matches new_role_code",
        "Creates pending_role_change on the employee record with case_id, template, and effective_date",
        "Returns case ID (CASE-ACCT-NNNNNN)",
    ],
    "schedule_access_review": [
        "Requires session.otp_auth = True",
        "Generates review ID (ARVW-NNNNNN) from the case_id",
        "Attaches access_review_id and review_date to the employee's pending_role_change",
        "Intended for ~90 days after the permission change effective date",
    ],
    # --- Flow 18: Access Removal ---
    "get_offboarding_record": [
        "Requires session.manager_auth = True",
        "Looks up employee's offboarding_record",
        "Returns error if no offboarding record exists (HR must initiate first)",
        "Returns full offboarding record (deep copy)",
    ],
    "submit_access_removal": [
        "Requires session.manager_auth = True AND session.otp_auth = True",
        "Validates departing employee is in the manager's direct_reports",
        "Checks offboarding_record exists on the departing employee",
        "Clears system_accounts and group_memberships, sets employment_status = 'terminated'",
        "For staged scope: preserves email for 30 days past last_working_day",
        "Updates offboarding_record with access_removed = True and case ID",
    ],
    "initiate_asset_recovery": [
        "Requires session.manager_auth = True",
        "Finds all assets assigned to departing_employee_id",
        "Generates recovery ID (RECV-NNNNNN) from case_id",
        "Returns list of assets to recover and the chosen recovery_method",
    ],
    # --- System ---
    "transfer_to_agent": [
        "No specific auth check required (params validated only)",
        "Generates transfer ID from employee_id and call_index",
        "Returns transfer_reason, estimated wait time, and transfer ID",
    ],
}

TOOL_CATEGORIES: dict[str, list[str]] = {
    "Auth": [
        "verify_employee_auth",
        "initiate_otp_auth",
        "verify_otp_auth",
        "verify_manager_auth",
    ],
    "Shared Lookups": [
        "get_employee_record",
        "get_asset_record",
        "get_employee_assets",
    ],
    "Resolving Issues (Flows 1-4)": [
        "get_troubleshooting_guide",
        "attempt_account_unlock",
        "attempt_password_reset",
        "create_incident_ticket",
        "assign_sla_tier",
        "check_existing_outage",
        "add_affected_user",
        "link_known_error",
        "schedule_field_dispatch",
        "attach_diagnostic_log",
    ],
    "Hardware Requests (Flows 5-6)": [
        "check_hardware_entitlement",
        "submit_hardware_request",
        "initiate_asset_return",
        "verify_cost_center_budget",
    ],
    "Software Requests (Flows 7-10)": [
        "get_application_details",
        "submit_access_request",
        "route_approval_workflow",
        "get_license_catalog_item",
        "validate_cost_center",
        "submit_license_request",
        "submit_temporary_license",
        "get_employee_licenses",
        "check_renewal_eligibility",
        "submit_license_renewal",
    ],
    "Facilities Requests (Flows 11-14)": [
        "check_space_availability",
        "submit_desk_assignment",
        "check_parking_availability",
        "submit_parking_assignment",
        "check_ergonomic_assessment",
        "submit_equipment_request",
        "check_room_availability",
        "submit_room_booking",
        "send_calendar_invite",
    ],
    "Accounts & Access (Flows 15-18)": [
        "lookup_new_hire",
        "check_existing_accounts",
        "provision_new_account",
        "get_group_memberships",
        "get_group_details",
        "submit_group_membership_change",
        "get_permission_templates",
        "submit_permission_change",
        "schedule_access_review",
        "get_offboarding_record",
        "submit_access_removal",
        "initiate_asset_recovery",
    ],
    "System": [
        "transfer_to_agent",
    ],
}

# ============================================================================
# Data loading
# ============================================================================

_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data
def load_flows_markdown() -> str | None:
    p = _ROOT / "data" / "itsm_flows.md"
    if not p.exists():
        return None
    return p.read_text()


@st.cache_data
def load_agent_instructions() -> str | None:
    p = _ROOT / "configs" / "agents" / "itsm_agent.yaml"
    if not p.exists():
        return None
    cfg = yaml.safe_load(p.read_text())
    return cfg.get("instructions", "")


@st.cache_data
def load_scenario_database() -> dict | None:
    p = _ROOT / "data" / "sample_itsm_scenario_database.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_tool_source() -> str:
    p = _ROOT / "src" / "eva" / "assistant" / "tools" / "itsm_tools.py"
    return p.read_text()


@st.cache_data
def load_params_source() -> str:
    p = _ROOT / "src" / "eva" / "assistant" / "tools" / "itsm_params.py"
    return p.read_text()


def extract_tool_functions(source: str) -> dict[str, str]:
    """Split tool source into {function_name: source_code} for public functions."""
    # Match each top-level function definition up to the next one or end of file
    pattern = r"^(def (\w+)\(.*?)(?=\ndef |\Z)"
    matches = re.findall(pattern, source, re.DOTALL | re.MULTILINE)
    return {name: code.rstrip() for code, name in matches if not name.startswith("_")}


def extract_enums(source: str) -> dict[str, list[str]]:
    """Extract enum class names and their values from params source."""
    enums: dict[str, list[str]] = {}
    current_enum = None
    for line in source.splitlines():
        m = re.match(r"^class (\w+)\(StrEnum\):", line)
        if m:
            current_enum = m.group(1)
            enums[current_enum] = []
            continue
        if current_enum:
            vm = re.match(r'^\s+\w+\s*=\s*"([^"]+)"', line)
            if vm:
                enums[current_enum].append(vm.group(1))
            elif line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                current_enum = None
    return enums


def extract_param_models(source: str) -> dict[str, str]:
    """Extract param model classes and their source code."""
    pattern = r"^(class (\w+Params)\(BaseModel\):.*?)(?=\nclass |\n# |\Z)"
    matches = re.findall(pattern, source, re.DOTALL | re.MULTILINE)
    return {name: code.rstrip() for code, name in matches}


# Map tool names to their param model class names
TOOL_TO_PARAMS: dict[str, str] = {
    "verify_employee_auth": "VerifyEmployeeAuthParams",
    "initiate_otp_auth": "InitiateOtpAuthParams",
    "verify_otp_auth": "VerifyOtpAuthParams",
    "verify_manager_auth": "VerifyManagerAuthParams",
    "get_employee_record": "GetEmployeeRecordParams",
    "get_asset_record": "GetAssetRecordParams",
    "get_employee_assets": "GetEmployeeAssetsParams",
    "get_troubleshooting_guide": "GetTroubleshootingGuideParams",
    "attempt_account_unlock": "AttemptAccountUnlockParams",
    "attempt_password_reset": "AttemptPasswordResetParams",
    "create_incident_ticket": "CreateIncidentTicketParams",
    "assign_sla_tier": "AssignSLATierParams",
    "check_existing_outage": "CheckExistingOutageParams",
    "add_affected_user": "AddAffectedUserParams",
    "link_known_error": "LinkKnownErrorParams",
    "schedule_field_dispatch": "ScheduleFieldDispatchParams",
    "attach_diagnostic_log": "AttachDiagnosticLogParams",
    "check_hardware_entitlement": "CheckHardwareEntitlementParams",
    "submit_hardware_request": "SubmitHardwareRequestParams",
    "initiate_asset_return": "InitiateAssetReturnParams",
    "verify_cost_center_budget": "VerifyCostCenterBudgetParams",
    "get_application_details": "GetApplicationDetailsParams",
    "submit_access_request": "SubmitAccessRequestParams",
    "route_approval_workflow": "RouteApprovalWorkflowParams",
    "get_license_catalog_item": "GetLicenseCatalogItemParams",
    "validate_cost_center": "ValidateCostCenterParams",
    "submit_license_request": "SubmitLicenseRequestParams",
    "submit_temporary_license": "SubmitTemporaryLicenseParams",
    "get_employee_licenses": "GetEmployeeLicensesParams",
    "check_renewal_eligibility": "CheckRenewalEligibilityParams",
    "submit_license_renewal": "SubmitLicenseRenewalParams",
    "check_space_availability": "CheckSpaceAvailabilityParams",
    "submit_desk_assignment": "SubmitDeskAssignmentParams",
    "check_parking_availability": "CheckParkingAvailabilityParams",
    "submit_parking_assignment": "SubmitParkingAssignmentParams",
    "check_ergonomic_assessment": "CheckErgonomicAssessmentParams",
    "submit_equipment_request": "SubmitEquipmentRequestParams",
    "check_room_availability": "CheckRoomAvailabilityParams",
    "submit_room_booking": "SubmitRoomBookingParams",
    "send_calendar_invite": "SendCalendarInviteParams",
    "lookup_new_hire": "LookupNewHireParams",
    "check_existing_accounts": "CheckExistingAccountsParams",
    "provision_new_account": "ProvisionNewAccountParams",
    "get_group_memberships": "GetGroupMembershipsParams",
    "get_group_details": "GetGroupDetailsParams",
    "submit_group_membership_change": "SubmitGroupMembershipChangeParams",
    "get_permission_templates": "GetPermissionTemplatesParams",
    "submit_permission_change": "SubmitPermissionChangeParams",
    "schedule_access_review": "ScheduleAccessReviewParams",
    "get_offboarding_record": "GetOffboardingRecordParams",
    "submit_access_removal": "SubmitAccessRemovalParams",
    "initiate_asset_recovery": "InitiateAssetRecoveryParams",
    "transfer_to_agent": "TransferToAgentParams",
}

# ============================================================================
# Question definitions
# ============================================================================

QUESTIONS = [
    {
        "section": "Flows",
        "preamble": "Please review each of the high level flows to understand what the premise is and what the expected tool sequence is.",
        "items": [
            {
                "key": "flows_applicability",
                "label": "Any concerns about the applicability of the flows? Meaning are you concerned any are not representative of what happens in the real world, or are any of them illogical, etc?",
            },
            {
                "key": "flows_tool_sequence",
                "label": "Any concerns about the logic with the expected tool sequence for a flow? Maybe that it is not realistic, not logical, etc?",
            },
        ],
    },
    {
        "section": "Policies",
        "preamble": "Please review the full set of agent instructions/policies.",
        "items": [
            {
                "key": "policies_tool_support",
                "label": "For each policy, carefully check if it supports the expected modification tool sequence for the flow. Is the policy complete enough to support the flow?",
            },
            {
                "key": "policies_completeness",
                "label": "Is the policy complete enough to support the flow?",
            },
            {
                "key": "policies_contradictions",
                "label": "Any contradictions, or other issues with the policy?",
            },
        ],
    },
    {
        "section": "Tools",
        "preamble": "Please review the logic, parameters, and code for each tool.",
        "items": [
            {
                "key": "tools_param_overlap",
                "label": "Any errors with parameters that are too similar? Remember there should always be exactly 1 correct answer for any scenario, so when tool parameters for a modification tool have options that have overlap, are ambiguous, or are too similar to each other this is a problem. As is any parameter that is free text and modifies the scenario database.",
            },
            {
                "key": "tools_logic",
                "label": "Any tool logic and modifications to the scenario database that don't make sense? For example, in HR someone noticed that for malpractice update it overwrote the existing value instead of appending it which would be more logical.",
            },
            {
                "key": "tools_other",
                "label": "Any other concerns?",
            },
        ],
    },
    {
        "section": "Scenario Database",
        "preamble": "Please review the sample scenario database.",
        "items": [
            {
                "key": "db_structure",
                "label": "Any concerns about the structure?",
            },
        ],
    },
    {
        "section": "Additions",
        "preamble": "",
        "items": [
            {
                "key": "additions_new_flows",
                "label": "Any flows you think we should add? If so please describe what the premise would be, what tools would be involved, etc.",
            },
            {
                "key": "additions_test_cases",
                "label": "Any specific test cases within the existing flows we should try to make sure to add? For example, cases where the user's first pick is not available and they have to keep discussing with the agent to find an acceptable option.",
            },
        ],
    },
    {
        "section": "Other",
        "preamble": "",
        "items": [
            {
                "key": "other_comments",
                "label": "Any other comments, questions, or concerns?",
            },
        ],
    },
]

# ============================================================================
# Sidebar — review form
# ============================================================================

with st.sidebar:
    st.header("Review Form")

    with st.form("review_form"):
        reviewer_name = st.text_input("Your Name", placeholder="e.g. Jane Doe")

        responses: dict[str, str] = {}
        for section in QUESTIONS:
            with st.expander(section["section"], expanded=False):
                if section["preamble"]:
                    st.caption(section["preamble"])
                for item in section["items"]:
                    responses[item["key"]] = st.text_area(
                        item["label"],
                        key=item["key"],
                        height=400,
                    )

        submitted = st.form_submit_button("Submit Review", use_container_width=True)

    if submitted:
        if not reviewer_name.strip():
            st.error("Please enter your name before submitting.")
        else:
            review_data = {
                "reviewer_name": reviewer_name.strip(),
                "timestamp": datetime.now().isoformat(),
                "responses": responses,
            }
            safe_name = reviewer_name.strip().lower().replace(" ", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = _ROOT / "apps" / "itsm_reviews"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{safe_name}_{ts}.json"
            out_path.write_text(json.dumps(review_data, indent=2))
            st.success(f"Review saved: {out_path.name}")

# ============================================================================
# Main area
# ============================================================================

st.title("ITSM Voice Agent — Domain Review")
st.markdown(
    "Review the flows, policies, tools, and scenario database for the ITSM voice agent. "
    "Use the sidebar on the left to submit your feedback."
)

# --- 1. Flow Descriptions ---
with st.expander("1. Flow Descriptions", expanded=False):
    flows_md = load_flows_markdown()
    if flows_md:
        st.markdown(flows_md)
    else:
        st.warning("data/itsm_flows.md not found. Please copy the flows file into the data/ directory.")

# --- 2. Agent Instructions & Policies ---
with st.expander("2. Agent Instructions & Policies", expanded=False):
    instructions = load_agent_instructions()
    if instructions:
        st.markdown(instructions)
    else:
        st.warning("configs/agents/itsm_agent.yaml not found or has no instructions field.")

# --- 3. Tools & Parameters ---
with st.expander("3. Tools & Parameters", expanded=False):
    tool_source = load_tool_source()
    params_source = load_params_source()
    tool_functions = extract_tool_functions(tool_source)
    enums = extract_enums(params_source)
    param_models = extract_param_models(params_source)

    # Enum reference
    st.subheader("Enum Reference")
    st.markdown("These are the constrained value sets used across tool parameters:")
    for enum_name, values in enums.items():
        st.markdown(f"**{enum_name}:** `{'`, `'.join(values)}`")

    st.divider()

    # Tools by category
    tabs = st.tabs(list(TOOL_CATEGORIES.keys()))
    for tab, (category, tool_names) in zip(tabs, TOOL_CATEGORIES.items()):
        with tab:
            for tool_name in tool_names:
                st.markdown(f"#### `{tool_name}`")

                # Description
                desc = TOOL_DESCRIPTIONS.get(tool_name, [])
                if desc:
                    st.markdown("**What it does:**")
                    for bullet in desc:
                        st.markdown(f"- {bullet}")

                # Parameters model source
                param_class = TOOL_TO_PARAMS.get(tool_name)
                if param_class and param_class in param_models:
                    st.markdown("**Parameters:**")
                    st.code(param_models[param_class], language="python")

                # Tool source code
                if tool_name in tool_functions:
                    st.markdown("**Source code:**")
                    st.code(tool_functions[tool_name], language="python")

                st.divider()

# --- 4. Sample Scenario Database ---
with st.expander("4. Sample Scenario Database", expanded=False):
    db = load_scenario_database()
    if db:
        st.json(db)
    else:
        st.warning("data/sample_itsm_scenario_database.json not found.")
