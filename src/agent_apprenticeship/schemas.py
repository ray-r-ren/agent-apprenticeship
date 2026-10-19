from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

DictAny = dict[str, Any]
EnvironmentDomain = Literal[
    "terminal",
    "browser",
    "computer_use",
    "web",
    "search",
    "codebase",
    "software",
    "os",
    "android",
    "mcp",
    "api",
    "database",
    "spreadsheet",
    "file",
    "multimodal",
    "device",
    "robotics",
    "lab",
    "industrial",
    "physical",
    "mixed",
    "unknown",
]
InteractionSurface = Literal[
    "api",
    "mcp_tool",
    "mcp_resource",
    "mcp_prompt",
    "browser",
    "computer_use",
    "cli",
    "file",
    "codebase",
    "database",
    "spreadsheet",
    "search",
    "cloud_console",
    "saas_app",
    "async_job",
    "queue",
    "webhook",
    "scheduler",
    "messaging",
    "email",
    "collaboration",
    "document",
    "pdf",
    "table",
    "multimodal",
    "device",
    "industrial",
    "physical",
    "mixed",
    "other_tool",
    "other_surface",
    "unknown",
]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ArtifactRef(StrictModel):
    artifact_id: str
    task_id: str
    attempt_id: str | None = None
    artifact_kind: Literal["input","output","intermediate","log","trace","score","rubric","feedback","lesson","package_file","media","document","unknown"]
    artifact_role: Literal["task_input","worker_output","reviser_output","apprentice_output","reference","hidden_reference","grader_output","verifier_output","evaluator_output","system_log","trace_file","other"]
    workspace_path: str | None = None
    package_relative_path: str
    release_relative_path: str | None = None
    mime_type: str | None = None
    media_type: Literal["text","code","data","document","image","audio","video","archive","binary","unknown"]
    size_bytes: int | None = None
    content_hash: str | None = None
    secret_scan_ok: bool
    metadata_json: DictAny = Field(default_factory=dict)

class ActualOutputs(StrictModel):
    task_id: str
    attempt_id: str
    attempt_kind: str
    status: Literal["success","partial","failed","timeout","error"]
    output_summary: str
    primary_output_ref: str | None = None
    input_artifact_refs: list[str] = Field(default_factory=list)
    deliverable_refs: list[str] = Field(default_factory=list)
    final_message_ref: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    files_deleted: list[str] = Field(default_factory=list)
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    raw_log_refs: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    metadata_json: DictAny = Field(default_factory=dict)

class WorkflowTraceStep(StrictModel):
    step: int
    turn: int
    actor: str
    action: Literal["user_message","agent_step","output","error"]
    operation: Literal["plan","analyze","search","read","write","edit","execute","verify","download","install","ask_user","answer","select","grade","evaluate","revise","other"] | None = None
    tool: str | None = None
    environment_domain: EnvironmentDomain | None = None
    execution_mode: Literal["serial","parallel"] | None = None
    parallel_group: str | None = None
    observation: str | None = None
    input: str | None = None
    input_source: DictAny | None = None
    output: str | None = None
    state_change: str | None = None
    reasoning: str | None = None
    caused_by: list[int] | None = None
    causal_type: Literal["user_request","follow_up_user_request","answer_to_agent_question","execution_of_plan","dependency_on_tool_result","retry_after_failure","correction_response","approval_response","verification_of_prior_step","dependency_on_multiple_prior_steps","delegation_to_subagent","delegated_work","used_subagent_result","handoff_from_subagent","parallel_work","other"] | None = None
    causal_note: str | None = None
    alternatives_considered: str | None = None
    success: bool | None = None
    step_outcome: Literal["progress","neutral","blocked","failed","corrected","completed"] | None = None
    error_type: str | None = None
    error_message: str | None = None
    message_role: Literal["direct_request","answer_to_agent_question","correction","approval","clarification","selection","status_update","new_constraint","other"] | None = None
    feedback_type: Literal["correction","approval","clarification","new_instruction","other"] | None = None
    feedback_content: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    retry_of: int | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    actor_kind: Literal["user","apprentice_agent","mentor_model","verifier","grader","evaluator","subagent","tool","system","expert","unknown"] | None = None
    actor_role: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    subagent_id: str | None = None
    subagent_name: str | None = None
    subagent_role: str | None = None
    parent_agent_id: str | None = None
    delegation_id: str | None = None
    delegation_source_step: int | None = None
    handoff_reason: str | None = None
    handoff_payload_ref: str | None = None
    handoff_output_refs: list[str] = Field(default_factory=list)
    subagent_output_refs: list[str] = Field(default_factory=list)
    coordination_pattern: Literal["sequential","parallel","debate","planner_worker","reviewer_reviser","swarm","unknown"] | None = None
    communication_channel: Literal["trace","file","message","tool_call","api","shared_workspace","unknown"] | None = None
    depends_on_agent_outputs: list[str] = Field(default_factory=list)
    modality: Literal["text","image","screenshot","audio","video","pdf","spreadsheet","table","diagram","mixed","unknown"] | None = None
    modalities: list[str] = Field(default_factory=list)
    media_type: str | None = None
    mime_type: str | None = None
    artifact_role: Literal["input","output","evidence","intermediate","preview","transcript","ocr","annotation","verification"] | None = None
    artifact_hash: str | None = None
    artifact_preview_ref: str | None = None
    ocr_text_ref: str | None = None
    transcript_ref: str | None = None
    annotation_ref: str | None = None
    bbox_refs: list[str] = Field(default_factory=list)
    frame_refs: list[str] = Field(default_factory=list)
    timecode_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    sheet_refs: list[str] = Field(default_factory=list)
    derived_text_summary: str | None = None
    interaction_surface: InteractionSurface | None = None
    surface_type: str | None = None
    connected_surfaces: list[str] = Field(default_factory=list)
    resource_id: str | None = None
    resource_type: str | None = None
    object_id: str | None = None
    object_type: str | None = None
    request_id: str | None = None
    operation_id: str | None = None
    job_id: str | None = None
    event_id: str | None = None
    session_id: str | None = None
    transaction_id: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    status_before: str | None = None
    status_after: str | None = None
    state_before_ref: str | None = None
    state_after_ref: str | None = None
    raw_request_ref: str | None = None
    raw_response_ref: str | None = None
    error_code: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    verification_refs: list[str] = Field(default_factory=list)
    redaction_notes: list[str] = Field(default_factory=list)
    api_provider: str | None = None
    api_service: str | None = None
    api_endpoint: str | None = None
    api_method: str | None = None
    api_version: str | None = None
    http_status: int | None = None
    request_payload_ref: str | None = None
    response_payload_ref: str | None = None
    request_schema_ref: str | None = None
    response_schema_ref: str | None = None
    auth_mode: str | None = None
    rate_limit_state: str | None = None
    pagination_state: str | None = None
    retry_count: int | None = None
    mcp_server: str | None = None
    mcp_server_version: str | None = None
    mcp_capability_type: Literal["tool","resource","prompt"] | None = None
    mcp_tool_name: str | None = None
    mcp_resource_uri: str | None = None
    mcp_prompt_name: str | None = None
    mcp_arguments_ref: str | None = None
    mcp_result_ref: str | None = None
    mcp_schema_ref: str | None = None
    mcp_error_code: str | None = None
    mcp_error_message: str | None = None
    mcp_call_id: str | None = None
    database_type: str | None = None
    database_name: str | None = None
    schema_name: str | None = None
    table_name: str | None = None
    query_type: str | None = None
    query_ref: str | None = None
    query_hash: str | None = None
    row_count: int | None = None
    affected_row_count: int | None = None
    isolation_level: str | None = None
    migration_id: str | None = None
    explain_plan_ref: str | None = None
    before_snapshot_ref: str | None = None
    after_snapshot_ref: str | None = None
    rollback_ref: str | None = None
    db_error_code: str | None = None
    db_error_message: str | None = None
    workbook_ref: str | None = None
    sheet_name: str | None = None
    table_ref: str | None = None
    cell_range: str | None = None
    column_count: int | None = None
    formula_refs: list[str] = Field(default_factory=list)
    pivot_refs: list[str] = Field(default_factory=list)
    chart_refs: list[str] = Field(default_factory=list)
    data_validation_refs: list[str] = Field(default_factory=list)
    before_table_ref: str | None = None
    after_table_ref: str | None = None
    diff_ref: str | None = None
    calculation_mode: str | None = None
    spreadsheet_error: str | None = None
    command: str | None = None
    command_ref: str | None = None
    cwd_ref: str | None = None
    exit_code: int | None = None
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    file_path: str | None = None
    file_paths: list[str] = Field(default_factory=list)
    file_operation: str | None = None
    patch_ref: str | None = None
    git_branch: str | None = None
    git_commit_before: str | None = None
    git_commit_after: str | None = None
    test_command: str | None = None
    test_result_ref: str | None = None
    build_result_ref: str | None = None
    lint_result_ref: str | None = None
    dependency_changes_ref: str | None = None
    process_id: str | None = None
    timeout_seconds: int | None = None
    search_provider: str | None = None
    query: str | None = None
    query_ref: str | None = None
    search_scope: str | None = None
    filters: DictAny | None = None
    result_count: int | None = None
    selected_result_refs: list[str] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)
    ranking_signal: str | None = None
    search_time_range: str | None = None
    search_error: str | None = None
    ui_surface: Literal["browser","desktop_app","mobile_app","terminal","ide","cloud_console","file_manager","spreadsheet_app","unknown"] | None = None
    app_name: str | None = None
    app_version: str | None = None
    os_name: str | None = None
    device_type: str | None = None
    window_title: str | None = None
    page_title: str | None = None
    url_or_resource_id: str | None = None
    visible_state_ref: str | None = None
    screenshot_ref: str | None = None
    dom_snapshot_ref: str | None = None
    accessibility_tree_ref: str | None = None
    element_selector: str | None = None
    coordinates: str | None = None
    action_type: str | None = None
    input_text: str | None = None
    before_state_ref: str | None = None
    after_state_ref: str | None = None
    network_ref: str | None = None
    api_call_ref: str | None = None
    software_state_ref: str | None = None
    service_name: str | None = None
    workspace_id: str | None = None
    project_id: str | None = None
    tenant_id: str | None = None
    record_id: str | None = None
    ticket_id: str | None = None
    object_state_before_ref: str | None = None
    object_state_after_ref: str | None = None
    permission_context: str | None = None
    region: str | None = None
    environment_name: str | None = None
    deployment_id: str | None = None
    audit_log_ref: str | None = None
    change_request_ref: str | None = None
    queue_name: str | None = None
    message_id: str | None = None
    webhook_id: str | None = None
    callback_url_ref: str | None = None
    schedule_id: str | None = None
    trigger_type: str | None = None
    enqueue_time: str | None = None
    start_time: str | None = None
    completion_time: str | None = None
    delivery_status: str | None = None
    polling_status: str | None = None
    callback_status: str | None = None
    job_status_before: str | None = None
    job_status_after: str | None = None
    async_result_ref: str | None = None
    dead_letter_ref: str | None = None
    platform_name: str | None = None
    channel_id: str | None = None
    thread_id: str | None = None
    sender_role: str | None = None
    recipient_role: str | None = None
    message_type: str | None = None
    message_ref: str | None = None
    attachment_refs: list[str] = Field(default_factory=list)
    decision_refs: list[str] = Field(default_factory=list)
    approval_state: str | None = None
    followup_required: bool | None = None
    document_ref: str | None = None
    document_type: str | None = None
    page_refs: list[str] = Field(default_factory=list)
    section_refs: list[str] = Field(default_factory=list)
    text_extract_ref: str | None = None
    table_extract_refs: list[str] = Field(default_factory=list)
    figure_refs: list[str] = Field(default_factory=list)
    annotation_refs: list[str] = Field(default_factory=list)
    summary_ref: str | None = None
    document_diff_ref: str | None = None
    physical_environment_type: str | None = None
    device_id: str | None = None
    device_model: str | None = None
    sensor_readings_ref: str | None = None
    actuator_command: str | None = None
    control_mode: str | None = None
    physical_state_before_ref: str | None = None
    physical_state_after_ref: str | None = None
    telemetry_ref: str | None = None
    calibration_ref: str | None = None
    safety_state: Literal["normal","warning","fault","emergency_stop","unknown"] | None = None
    safety_limit_refs: list[str] = Field(default_factory=list)
    operator_override: bool | None = None
    human_supervisor_present: bool | None = None
    simulation_or_real_world: Literal["simulation","real_world","hybrid","unknown"] | None = None
    latency_ms: float | None = None
    tolerance: str | None = None
    rollback_action: str | None = None
    emergency_stop: bool | None = None
    compliance_refs: list[str] = Field(default_factory=list)
    cross_system_dependencies: list[str] = Field(default_factory=list)
    handoff_refs: list[str] = Field(default_factory=list)
    state_sync_refs: list[str] = Field(default_factory=list)
    consistency_checks: list[str] = Field(default_factory=list)
    system_of_record: str | None = None
    surface_capture_status: Literal["complete","partial","unavailable","not_applicable"] | None = None
    surface_capture_notes: str | None = None
    missing_surface_fields: list[str] = Field(default_factory=list)
    metadata_json: DictAny = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_step(self):
        if self.step < 1: raise ValueError("step must start at 1")
        if self.action == "user_message":
            required_null = ["operation","tool","execution_mode","observation","reasoning","success","step_outcome","output"]
            bad = [name for name in required_null if getattr(self, name) is not None]
            if self.actor != "user" or bad:
                raise ValueError(f"user_message invariants failed: {bad}")
        elif self.operation is None:
            raise ValueError("operation is required for non-user steps")
        return self

class WorkflowTrace(StrictModel):
    """Internal task/evaluation projection. Persist using ATIF I/O only."""

    @model_validator(mode="before")
    @classmethod
    def from_atif(cls, value):
        from .atif import to_workflow
        if isinstance(value, dict) and "agent" in value and "steps" in value:
            return to_workflow(value)
        return value

    schema_version: str = "ATIF-v1.8"
    trace_id: str
    collection_id: str | None = None
    run_id: str | None = None
    package_id: str | None = None
    bundle_id: str | None = None
    prior_trace_id: str | None = None
    trace_mode: Literal["live","retraced","hybrid"]
    task: str
    task_id: str
    task_family_id: str | None = None
    attempt_id: str
    attempt_kind: Literal["baseline","revised","apprentice_without_lessons","apprentice_with_lessons","other"]
    attempt_status: Literal["completed","failed","blocked","fallback","partial"] | None = None
    agent_tools: list[str]
    started_at: str | None = None
    ended_at: str | None = None
    system_prompt: str | None = None
    system_prompt_hash: str | None = None
    skills: list[str] | None = None
    memory: str | None = None
    agent_config: DictAny | None = None
    environment_domain: EnvironmentDomain | None = None
    actual_outputs_ref: str | None = None
    input_artifact_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    deliverable_refs: list[str] = Field(default_factory=list)
    output_summary: str | None = None
    final_output_summary: str | None = None
    learning: str | None = None
    termination_reason: Literal["task_complete","verifier_passed","verifier_failed","max_iterations_reached","agent_blocked","timeout","error_unrecoverable","partial_then_stopped","provider_usage_limit","other"]
    steps: list[WorkflowTraceStep]
    actual_outputs: ActualOutputs | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    iteration_index: int | None = None
    previous_attempt_id: str | None = None
    revision_group_id: str | None = None
    completion_reason: str | None = None
    final_attempt_id: str | None = None
    preferred_attempt_id: str | None = None
    initial_attempt_id: str | None = None
    revision_attempt_ids: list[str] = Field(default_factory=list)
    metadata_json: DictAny = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trace(self):
        if not self.steps: raise ValueError("trace must have steps")
        nums = [s.step for s in self.steps]
        if nums != list(range(1, len(nums)+1)): raise ValueError("step numbers must be monotonic starting at 1")
        seen=set()
        for s in self.steps:
            if s.caused_by and any(c >= s.step or c < 1 for c in s.caused_by):
                raise ValueError("causality refs must point to earlier steps")
            seen.add(s.step)
        return self

class RawTaskRecord(StrictModel):
    raw_task_id: str
    source_kind: str
    source_url: str | None = None
    source_license: str | None = None
    raw_title: str
    raw_description: str
    raw_payload: DictAny = Field(default_factory=dict)
    task_id: str | None = None
    normalized_title: str | None = None
    normalized_instruction: str | None = None
    input_artifact_refs: list[str] = Field(default_factory=list)
    created_at: str | None = None
    metadata_json: DictAny = Field(default_factory=dict)
    normalized_domain: str | None = None
    normalized_subdomain: str | None = None
    apprenticeship_role: str | None = None
    task_family: str | None = None
    expected_deliverable: str | None = None
    expected_economic_value: str | None = None
    expected_economic_value_for_agent_apprentice: str | None = None
    expected_pay: str | None = None
    expected_apprentice_pay: str | None = None
    source_url_or_ref: str | None = None
    difficulty_tier: Literal["easy","medium","hard","expert"] | None = None
    needs_expert_review: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_task_sheet(cls, data):
        if not isinstance(data, dict):
            return data
        d=dict(data)
        payload=dict(d.get('raw_payload') or {})
        if 'task_id' in d and 'raw_task_id' not in d:
            d['raw_task_id']=d['task_id']
        if 'normalized_title' in d and 'raw_title' not in d:
            d['raw_title']=d['normalized_title']
        if 'normalized_instruction' in d and 'raw_description' not in d:
            d['raw_description']=d['normalized_instruction']
        if 'source_url_or_ref' in d and 'source_url' not in d:
            d['source_url']=d['source_url_or_ref']
        if 'expected_pay' in d and 'expected_economic_value' not in d:
            d['expected_economic_value'] = d['expected_pay']
        if 'expected_apprentice_pay' in d and 'expected_economic_value_for_agent_apprentice' not in d:
            d['expected_economic_value_for_agent_apprentice'] = d['expected_apprentice_pay']
        for key in ['normalized_domain','normalized_subdomain','apprenticeship_role','task_family','expected_deliverable','expected_economic_value','expected_economic_value_for_agent_apprentice','expected_pay','expected_apprentice_pay','difficulty_tier','needs_expert_review']:
            if key in d and key not in payload:
                payload[key]=d[key]
        d.setdefault('source_kind', d.get('source_kind') or 'manual_seed')
        d.setdefault('raw_title', d.get('raw_task_id','untitled_task'))
        d.setdefault('raw_description', d.get('expected_deliverable') or '')
        d['raw_payload']=payload
        return d

class TaskIntakeSpec(StrictModel):
    task_id: str; normalized_title: str; normalized_instruction: str; domain: str
    subdomain: str | None = None; professional_role: str | None = None; apprenticeship_role: str | None = None; task_family: str | None = None; expected_economic_value: str | None = None; expected_economic_value_for_agent_apprentice: str | None = None; expected_pay: str | None = None; expected_apprentice_pay: str | None = None; workflow_type: str
    skill_targets: list[str] = Field(default_factory=list)
    difficulty_tier: Literal["easy","medium","hard","expert"]
    expected_human_deliverable: str; expected_agent_deliverable: str
    input_requirements: list[str] = Field(default_factory=list); output_requirements: list[str] = Field(default_factory=list)
    required_context: list[str] = Field(default_factory=list); assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list); allowed_tools: list[str] = Field(default_factory=list); disallowed_tools: list[str] = Field(default_factory=list)
    privacy_classification: Literal["public","synthetic","sensitive_possible","contains_pii","unknown"]
    license: str | None = None; allowed_use: str | None = None
    rubricability_score: float; verifiability_score: float; artifactability_score: float
    needs_expert_review: bool; metadata_json: DictAny = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_economic_fields(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if d.get('expected_pay') is not None and d.get('expected_economic_value') is None:
            d['expected_economic_value'] = d.get('expected_pay')
        if d.get('expected_apprentice_pay') is not None and d.get('expected_economic_value_for_agent_apprentice') is None:
            d['expected_economic_value_for_agent_apprentice'] = d.get('expected_apprentice_pay')
        if d.get('expected_economic_value') is not None and d.get('expected_pay') is None:
            d['expected_pay'] = d.get('expected_economic_value')
        if d.get('expected_economic_value_for_agent_apprentice') is not None and d.get('expected_apprentice_pay') is None:
            d['expected_apprentice_pay'] = d.get('expected_economic_value_for_agent_apprentice')
        return d

class TaskIntakeQualityReport(StrictModel):
    task_id: str; instruction_clarity_score: float; input_completeness_score: float; output_contract_score: float
    rubricability_score: float; verifiability_score: float; artifactability_score: float; privacy_risk_score: float; license_risk_score: float; ambiguity_score: float; overall_intake_quality_score: float
    quality_flags: list[str] = Field(default_factory=list); blockers: list[str] = Field(default_factory=list); recommended_fix: str | None = None; metadata_json: DictAny = Field(default_factory=dict)

class RubricItem(StrictModel):
    rubric_item_id: str; criterion_name: str; criterion_description: str; weight: float; score_min: float; score_max: float; pass_threshold: float
    observable_evidence: list[str]; required_artifacts: list[str]
    scoring_method: Literal["llm_rubric_judge","deterministic","schema_match","regex","unit_test","hybrid","human_future","structural_guardrail"]
    worker_visible: bool; verifier_only: bool; hidden_reference_required: bool
    failure_modes: list[str] = Field(default_factory=list); partial_credit_rules: list[str] = Field(default_factory=list); edge_cases: list[str] = Field(default_factory=list); anti_cheat_notes: list[str] = Field(default_factory=list); metadata_json: DictAny = Field(default_factory=dict)

class RubricSpec(StrictModel):
    rubric_id: str; task_id: str; task_family_id: str | None = None; rubric_version: str; rubric_items: list[RubricItem]; total_weight: float; pass_threshold: float
    worker_visible_rubric_ref: str; verifier_private_rubric_ref: str; hidden_reference_policy: str
    scoring_aggregation: Literal["weighted_sum","sum","all_required","custom"]
    required_artifacts: list[str]; disqualifying_errors: list[str] = Field(default_factory=list); partial_credit_allowed: bool
    grader_kind: Literal["llm_rubric_judge","deterministic","hybrid","human_future","structural_guardrail","unavailable"]
    rubric_generation_source: Literal["agent_assisted","family_template","task_specific_agent_draft","expert_override","deterministic_seed"]
    rubric_generation_agent_provider: str | None = None; rubric_generation_agent_model: str | None = None; rubric_generation_confidence: float | None = None; metadata_json: DictAny = Field(default_factory=dict)
    @model_validator(mode="after")
    def valid_rubric(self):
        if not self.rubric_items: raise ValueError("rubric_items required")
        if not (abs(sum(i.weight for i in self.rubric_items)-1.0)<1e-6 or abs(sum(i.weight for i in self.rubric_items)-100)<1e-6): raise ValueError("weights must sum to 1.0 or 100")
        if any(not i.observable_evidence or not i.required_artifacts for i in self.rubric_items): raise ValueError("observable evidence and artifacts required")
        return self

class RubricQualityReport(StrictModel):
    rubric_id: str; task_id: str; criteria_count: int; total_weight: float; weights_sum_valid: bool; has_observable_evidence: bool; has_required_artifacts: bool; has_partial_credit_rules: bool; has_disqualifying_errors: bool; has_hidden_reference_policy: bool; has_worker_visible_view: bool; has_verifier_private_view: bool; ambiguous_criteria_count: int; unverifiable_criteria_count: int; rubric_quality_score: float; quality_flags: list[str] = Field(default_factory=list); blockers: list[str] = Field(default_factory=list); metadata_json: DictAny = Field(default_factory=dict)

class RubricItemScore(StrictModel):
    rubric_item_id: str; criterion_name: str; score: float; max_score: float; passed: bool; evidence_refs: list[str] = Field(default_factory=list); failure_mode: str | None = None; notes: str | None = None; confidence: float | None = None; artifact_presence_ok: bool | None = None; semantic_correctness_score: float | None = None; reasoning_summary: str | None = None; improvement_suggestion: str | None = None
class GraderResult(StrictModel):
    grader_result_id: str; task_id: str; attempt_id: str; attempt_kind: str; rubric_id: str; grader_kind: Literal["llm_rubric_judge","llm","model","deterministic","hybrid","human_future","structural_guardrail","unavailable"]; score_source: Literal["llm","llm_semantic","model_judged","deterministic","deterministic_artifact_contract","deterministic_fallback","hybrid","human_future","structural_guardrail","unavailable"]; score: float; max_score: float; passed: bool; rubric_item_scores: list[RubricItemScore]; failed_criteria: list[str]; passed_criteria: list[str]; evidence_refs: list[str]; confidence: float; reasoning_summary: str | None = None; limitations: list[str] = Field(default_factory=list); hidden_reference_used: bool; hidden_reference_leaked: bool; artifact_contract_score: float | None = None; semantic_score: float | None = None; model_score: float | None = None; legacy_semantic_score: float | None = None; legacy_score_source: str | None = None; final_score: float | None = None; model: str | None = None; provider: str | None = None; deterministic_precheck_ref: str | None = None; llm_prompt_ref_internal: str | None = None; llm_response_ref_internal: str | None = None; public_prompt_hash: str | None = None; public_response_summary: str | None = None; score_reliability: Literal["verified","unverified","needs_review","failed_verification"] | None = None; verifier_status: str | None = None; verifier_confidence: float | None = None; verifier_issue_count: int | None = None; verifier_issues_summary: str | None = None; metadata_json: DictAny = Field(default_factory=dict)
class VerifierResult(StrictModel):
    verifier_result_id: str; task_id: str; attempt_id: str; attempt_kind: str; grader_result_id: str | None = None; verification_status: Literal["verified","partially_verified","failed","not_run"]; artifact_contract_ok: bool; evidence_grounding_ok: bool; score_consistency_ok: bool; hidden_reference_leaked: bool; issues: list[str] = Field(default_factory=list); confidence: float; verifier_notes: str | None = None; semantic_evidence_grounding_ok: bool | None = None; unsupported_claims: list[str] = Field(default_factory=list); leakage_check_ok: bool | None = None; model: str | None = None; provider: str | None = None; metadata_json: DictAny = Field(default_factory=dict)
class EvaluatorFeedback(StrictModel):
    feedback_id: str; task_id: str; attempt_id: str; target_actor: Literal["worker","reviser","apprentice"]; feedback_type: Literal["criteria_failure","artifact_missing","format_error","logic_error","tool_error","quality_gap","strategy_gap","safety_or_privacy","other"]; failed_rubric_items: list[str]; evidence_refs: list[str]; artifact_refs: list[str]; feedback_summary: str; actionable_feedback: list[str]; suggested_revision: str; revision_priority: Literal["low","medium","high"]; confidence: float; hidden_reference_used: bool; hidden_reference_leaked: bool; failed_or_weak_rubric_items: list[str] = Field(default_factory=list); artifact_specific_comments: list[str] = Field(default_factory=list); trace_specific_comments: list[str] = Field(default_factory=list); revision_plan: str | None = None; model: str | None = None; provider: str | None = None; metadata_json: DictAny = Field(default_factory=dict)
class RevisionPlan(StrictModel):
    revision_plan_id: str; task_id: str; source_attempt_id: str; target_attempt_id: str; revision_kind: Literal["local_fix","strategy_shift","tool_change","decomposition_change","artifact_rebuild","format_repair","other"]; revision_reason: str; failed_rubric_items: list[str]; planned_changes: list[str]; expected_score_improvement: float | None = None; risk_of_regression: Literal["low","medium","high"]; uses_evaluator_feedback: bool; metadata_json: DictAny = Field(default_factory=dict)
class HillclimbResult(StrictModel):
    hillclimb_id: str; task_id: str; baseline_attempt_id: str; revised_attempt_id: str; baseline_score: float; revised_score: float; revision_score_delta: float; baseline_passed: bool; revised_passed: bool; failed_criteria_before: list[str]; failed_criteria_after: list[str]; criteria_improved: list[str]; criteria_regressed: list[str]; artifact_completeness_before: float; artifact_completeness_after: float; artifact_completeness_delta: float; regression_count: int; improvement_kind: Literal["score_delta","pass_delta","criteria_delta","artifact_delta","none","regression"]; hillclimb_evidence_strength: Literal["observed_improvement","no_observed_improvement","regression_observed"]; revision_success: bool; metadata_json: DictAny = Field(default_factory=dict)
class LessonPack(StrictModel):
    lesson_id: str; task_id: str; source_attempt_ids: list[str]; lesson_summary: str; strategy_lessons: list[str]; common_failure_modes: list[str]; rubric_reminders: list[str]; artifact_requirements: list[str]; verifier_feedback_summary: str; hidden_reference_leaked: bool; metadata_json: DictAny = Field(default_factory=dict)
class TrainingSignal(StrictModel):
    signal_id: str; task_id: str; signal_type: Literal["rollout","process_supervision","verifier_training","reward_modeling","revision_preference","lesson_transfer"]; source_attempt_ids: list[str]; baseline_score: float | None = None; revised_score: float | None = None; score_delta: float | None = None; criteria_improved: list[str]=Field(default_factory=list); criteria_regressed: list[str]=Field(default_factory=list); failed_criteria_before: list[str]=Field(default_factory=list); failed_criteria_after: list[str]=Field(default_factory=list); feedback_ref: str | None = None; revision_plan_ref: str | None = None; grader_result_refs: list[str]=Field(default_factory=list); verifier_result_refs: list[str]=Field(default_factory=list); trace_refs: list[str]=Field(default_factory=list); artifact_refs: list[str]=Field(default_factory=list); score_source: str; grader_kind: str; confidence: float | None = None; training_use_cases: list[str]=Field(default_factory=list); limitations: list[str]=Field(default_factory=list); metadata_json: DictAny=Field(default_factory=dict)
class ProcessSupervisionExample(StrictModel):
    example_id: str; task_id: str; attempt_id: str; attempt_kind: str | None = None; trace_id: str; step: int; actor: str; action: str; operation: str | None = None; tool: str | None = None; observation: str | None = None; input: str | None = None; output: str | None = None; state_change: str | None = None; reasoning: str | None = None; caused_by: list[int] | None = None; causal_type: str | None = None; success: bool | None = None; step_outcome: str | None = None; step_quality_label: Literal["positive","neutral","negative","unknown"]; local_reward: float | None = None; failure_mode: str | None = None; grader_feedback: str | None = None; verifier_feedback: str | None = None; evaluator_feedback: str | None = None; revision_reason: str | None = None; final_outcome_score: float | None = None; label_source: Literal["grader","verifier","evaluator","heuristic","human_future","none"]; label: str = ""; metadata_json: DictAny = Field(default_factory=dict)
class RewardModelingExample(StrictModel):
    example_id: str; task_id: str; attempt_id: str; rubric_ref: str; output_refs: list[str]; attempt_summary: str; rubric_item_scores: list[RubricItemScore]; final_score: float; passed: bool; failure_modes: list[str]; grader_notes: str | None = None; evidence_refs: list[str]; score_source: str; grader_kind: str; confidence: float; score_reliability: str | None = None; verifier_status: str | None = None; verifier_confidence: float | None = None; verifier_issue_count: int | None = None; verifier_issues_summary: str | None = None; metadata_json: DictAny=Field(default_factory=dict)
class RevisionPreferencePair(StrictModel):
    pair_id: str; task_id: str; rubric_ref: str; baseline_attempt_ref: str; revised_attempt_ref: str; chosen_attempt_id: str; rejected_attempt_id: str; baseline_score: float; revised_score: float; score_delta: float; criteria_improved: list[str]; criteria_regressed: list[str]; preference_reason: str; score_source: str; grader_kind: str; confidence: float; metadata_json: DictAny=Field(default_factory=dict)

class TDOJudgeResult(StrictModel):
    tdo_round: int
    tdo_judge_source: Literal["mentor_model", "apprentice_self_judge"]
    mentor_audited: bool
    verdict: Literal["accept", "improve", "reject"]
    weak_pattern: str | None = None
    strong_pattern: str | None = None
    gap_interpretation: str | None = None
    rubric_concerns: list[str] = Field(default_factory=list)
    grounding_concerns: list[str] = Field(default_factory=list)
    data_quality_concerns: list[str] = Field(default_factory=list)
    grpo_suitability: Literal["high", "medium", "low", "unknown"]
    learning_signal_quality: Literal["high", "medium", "low", "unknown"]
    reuse_value: Literal["high", "medium", "low"]
    quality_score: float
    fidelity_score: float
    difficulty_score: float
    learnability_score: float
    reuse_value_score: float
    verdict_reason: str
    suggestion_for_generator: str | None = None
    metadata_json: DictAny = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_suggestion(self):
        if self.verdict in {"improve", "reject"} and not self.suggestion_for_generator:
            raise ValueError("suggestion_for_generator is required on improve/reject")
        return self

class TDOReport(StrictModel):
    tdo_id: str
    source_package_id: str
    source_task_id: str | None = None
    created_at: str
    generator_mode: str
    tdo_judge_source: Literal["mentor_model", "apprentice_self_judge"]
    mentor_audited: bool
    rounds_run: int
    stop_reason: str
    tdo_status: Literal["completed", "failed", "disabled", "partial"]
    rows_generated_by_type: DictAny = Field(default_factory=dict)
    rows_accepted_by_type: DictAny = Field(default_factory=dict)
    rows_rejected_by_type: DictAny = Field(default_factory=dict)
    average_quality_score: float | None = None
    average_fidelity_score: float | None = None
    average_difficulty_score: float | None = None
    average_learnability_score: float | None = None
    average_reuse_value_score: float | None = None
    schema_validation_passed: bool
    sanitization_passed: bool
    weak_strong_proxy_available: bool
    average_solver_gap_proxy: float | None = None
    grpo_suitability_distribution: DictAny = Field(default_factory=dict)
    omitted_outputs: list[str] = Field(default_factory=list)
    not_generated_counts: DictAny = Field(default_factory=dict)
    generation_notes: list[str] = Field(default_factory=list)
    recommended_uses: list[str] = Field(default_factory=list)
    metadata_json: DictAny = Field(default_factory=dict)

# Public trace models use the Harbor interchange contract.
from .atif_models import Trajectory as AgentTrace, Step as AgentTraceStep
