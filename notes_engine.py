from llama_index.core.tools import FunctionTool
from pathlib import Path
import os

note_path = Path(__file__).parent / 'data/notes.txt'


def save_notes(note):
    """Save a note to a text file"""

    if not os.path.exists(note_path):
        open(note_path, "w")

    with open(note_path, "a") as fw:
        fw.writelines([f'{note}\n'])

    return "notes successfully saved"


note_engine = FunctionTool.from_defaults(
    fn=save_notes,
    name="notes_saver",
    description="this tool can save a text based note to a file for the user",
)