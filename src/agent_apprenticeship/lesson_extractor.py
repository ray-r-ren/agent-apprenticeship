from __future__ import annotations
from .schemas import LessonPack, HillclimbResult

def extract_lesson(h: HillclimbResult) -> LessonPack:
    return LessonPack(lesson_id=f'lesson_{h.task_id}', task_id=h.task_id, source_attempt_ids=[h.baseline_attempt_id,h.revised_attempt_id], lesson_summary='Use verifier and grader feedback to repair missing artifacts before revision.', strategy_lessons=['Map every output requirement to a concrete file.'], common_failure_modes=h.failed_criteria_before, rubric_reminders=h.failed_criteria_after, artifact_requirements=['Preserve package-relative artifact paths.'], verifier_feedback_summary=h.hillclimb_evidence_strength, hidden_reference_leaked=False, metadata_json={})
