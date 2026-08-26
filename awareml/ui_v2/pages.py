from __future__ import annotations

from .pages_core import command_center_page, decision_space_page, run_studio_v2_page
from .pages_observatory import streaming_observatory_page, responsible_ai_page
from .pages_copilot import copilot_workspace_page
from .pages_faithfulness import faithfulness_lab_page
from .pages_export import export_center_page
from .pages_advanced import advanced_labs_page


PAGE_REGISTRY_V2 = {
    "Command Center": command_center_page,
    "Run Studio": run_studio_v2_page,
    "3D Decision Space": decision_space_page,
    "Streaming Observatory": streaming_observatory_page,
    "Responsible AI": responsible_ai_page,
    "Copilot Workspace": copilot_workspace_page,
    "Faithfulness Lab": faithfulness_lab_page,
    "Export Center": export_center_page,
    "Advanced Labs": advanced_labs_page,
}
