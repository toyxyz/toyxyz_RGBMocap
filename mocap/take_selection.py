"""Saved-take labels and readiness shared by the People and Export controls."""
from dataclasses import dataclass
import json
from pathlib import Path

from .motion_tools import RESULTS, base_result, result_sources, result_is_current
from .person_names import export_stem


@dataclass(frozen=True)
class TakeChoice:
    person_id: int
    person_name: str
    number: int
    total: int
    first: int
    end: int
    project: Path
    source: Path | None
    issue: 'TakeIssue | None'

    @property
    def status(self):
        return 'Ready' if self.issue is None else self.issue.status

    @property
    def frames(self):
        return f'Frames {self.first + 1}–{self.end}'

    @property
    def stem(self):
        name = export_stem(self.person_name)
        return f'{name}_take_{self.number:02d}' if self.total > 1 else name

    @property
    def label(self):
        return f'{self.person_name}, Take {self.number} ({self.frames.lower()})'


@dataclass(frozen=True)
class TakeIssue:
    status: str
    reason: str
    action: str

    def report(self, label):
        return f'{label}\nReason: {self.reason}\nWhat to do: {self.action}'


def correction_reason(project, kind):
    from .correction_settings import correction_options, effective_options
    from .foot_contacts import contact_revision
    from .result_artifacts import pending_path
    from .scene_floor import receipt_matches

    if pending_path(project, kind, 'correct').is_file():
        return 'A previous Motion Correction run did not finish.'
    path=project/f'{kind}_correction.json'
    if not path.is_file():
        return 'This take has no saved correction for the selected motion.'
    try:
        state=json.loads((project/'project.json').read_text(encoding='utf-8'))
        report=json.loads(path.read_text(encoding='utf-8'))
        if correction_options(report.get('options',{}))!=effective_options(project,state):
            return 'The saved motion was corrected with different correction or floor settings.'
        if not receipt_matches(project,state,report.get('shared_floor')):
            return 'The saved correction uses an older shared scene floor or camera placement.'
        if report.get('contact_revision')!=contact_revision(project,base_result(kind)):
            return 'The contact-lock timings changed after the last correction.'
        if not (project/RESULTS[kind][0]).is_file():
            return 'The corrected motion file is missing.'
        if 'fps' in report and report['fps']!=state.get('fps',30):
            return 'The frame rate was changed after the last correction.'
    except (OSError,ValueError,KeyError,TypeError):
        return 'The saved correction data could not be read or verified.'
    return 'The saved correction no longer matches this take’s motion or body model.'


def diagnose_take(project, kind, pending_contacts=False):
    from .gemx_result import is_project
    from .result_artifacts import artifacts_current, pending_path

    if pending_contacts:
        return TakeIssue('Pending contact edits', 'There are contact-lock edits that have not finished saving.',
                         'Finish saving the contact edits, then click Apply Correction.')
    if is_project(project):
        if result_is_current(project,'initial'):
            return TakeIssue('Apply Correction', 'The saved SOMA correction is missing or no longer matches this take.',
                             'Select the take and click Apply Correction.')
        return TakeIssue('Reconstruct 3D', 'The saved SOMA motion is missing, incomplete, or out of date.',
                         'Select the take and click Reconstruct 3D.')

    if any((project/name).is_file() for name in ('flowhmr_pending.json','exmr_pending.json')):
        return TakeIssue('Reconstruct 3D', 'A previous 3D reconstruction did not finish.',
                         'Select the take and click Reconstruct 3D.')
    if not (project/RESULTS['initial'][0]).is_file():
        return TakeIssue('Reconstruct 3D', 'This take has no saved 3D body motion.',
                         'Select the take and click Reconstruct 3D.')
    bases=result_sources(base_result(kind))
    if any(not (project/RESULTS[base][0]).is_file() for base in bases):
        return TakeIssue('Reconstruct 3D', 'The take has body motion, but the selected refinement result is missing.',
                         'Choose an available motion result or run Reconstruct 3D with the intended refinement options.')
    if (project/'flow_prediction.pt').is_file():
        from .flow_placement import needs_update
        if needs_update(project):
            reason=('A previous FlowHMR placement update did not finish.' if (project/'flow_world_pending.json').is_file()
                    else 'The saved FlowHMR placement no longer matches its camera or motion inputs.')
            return TakeIssue('Prepare Preview', reason,
                             'Click Prepare Preview. Reconstruct 3D if the app reports an invalid prediction.')
    if kind.startswith('corrected_') and not result_is_current(project,kind):
        if all(result_is_current(project,base) for base in bases):
            return TakeIssue('Apply Correction', correction_reason(project,kind),
                             'Select the take and click Apply Correction.')
    if pending_path(project,kind,'export').is_file() or (project/f'{kind}_mixamo_pending.json').is_file():
        reason='A previous preview or export-skeleton preparation did not finish.'
    elif not result_is_current(project,base_result(kind)):
        return TakeIssue('Reconstruct 3D', 'The selected source motion no longer matches its inputs or model.',
                         'Reconstruct the take, or choose another available motion result.')
    elif not artifacts_current(project,kind):
        reason='The preview files are missing or no longer match the selected motion.'
    elif not (project/f'{kind}_mixamo.json').is_file():
        reason='The export skeleton has not been prepared for the selected motion.'
    else:
        reason='The saved export skeleton no longer matches the selected motion or preview.'
    return TakeIssue('Prepare Preview', reason, 'Select the take and click Prepare Preview.')
