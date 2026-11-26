"""Argument groups for operations shared by more than one model."""
from argparse import BooleanOptionalAction
from pathlib import Path

from .motion_tools import MOTION_TOOLS


def camera_refinement_flag(parser, help_text):
    parser.add_argument('--refine-camera-rotation', action=BooleanOptionalAction, help=help_text)


def option_parsers(commands, models, take_flags):
    def json_option(item):
        item.add_argument('--json', action='store_true')

    def dry(item):
        item.add_argument('--dry-run', action='store_true')

    def project_arg(item):
        item.add_argument('project', type=Path)
        item.add_argument('--model', choices=models, default='exmr')

    models_parser = commands.add_parser('models', help='List and manage installed model files.')
    actions = models_parser.add_subparsers(dest='action', required=True)
    listing = actions.add_parser('list')
    listing.add_argument('--workflow', choices=tuple(MOTION_TOOLS))
    listing.add_argument('--installed', action='store_true')
    json_option(listing)
    for action in ('import','download'):
        item = actions.add_parser(action)
        item.add_argument('key')
        if action == 'import':
            item.add_argument('file', type=Path)
            item.add_argument('--replace', action='store_true')
        json_option(item)
    removal = actions.add_parser('remove')
    removal.add_argument('keys', nargs='+')
    removal.add_argument('--yes', action='store_true')
    dry(removal)
    json_option(removal)

    faces = commands.add_parser('face', help='Track faces and manage neutral frames.')
    actions = faces.add_subparsers(dest='action', required=True)
    for action in ('track','inspect','neutral','reset-neutral'):
        item = actions.add_parser(action)
        project_arg(item)
        if action == 'track':
            item.add_argument('--person', type=int, action='append')
        elif action in ('neutral', 'reset-neutral'):
            item.add_argument('--person', type=int, default=1)
            if action == 'neutral':
                item.add_argument('--frame', type=int, required=True, help='1-based capture frame.')
        json_option(item)

    batch = commands.add_parser('batch', help='Run a versioned JSON queue sequentially.')
    batch.add_argument('file', type=Path)
    batch.add_argument('--keep-going', action='store_true')
    dry(batch)
    json_option(batch)

    prepare = commands.add_parser('prepare-preview', help='Refresh the saved rig and shared cameras.')
    project_arg(prepare)
    take_flags(prepare)
    dry(prepare)
    json_option(prepare)

    contacts = commands.add_parser('contacts', help='Read, set or reset saved contact-lock intervals.')
    actions = contacts.add_subparsers(dest='action', required=True)
    for action in ('list','set','reset'):
        item = actions.add_parser(action)
        project_arg(item)
        take_flags(item)
        if action == 'set':
            item.add_argument('ranges', type=Path, help='JSON array of two or four contact tracks.')
        json_option(item)

    reference = commands.add_parser('reference-fbx', help='Export the selected Live Link reference rig.')
    project_arg(reference)
    take_flags(reference)
    reference.add_argument('--output', type=Path, required=True)
    json_option(reference)

    character = commands.add_parser('character', help='Retarget a take and export animation.')
    project_arg(character)
    take_flags(character)
    character.add_argument('--asset', type=Path, required=True, help='FBX, VRM or PMX character.')
    character.add_argument('--output', type=Path, required=True, help='New FBX or VMD output.')
    character.add_argument('--no-mesh', action='store_true')
    dry(character)
    json_option(character)

    preview = commands.add_parser('preview', help='Render an MP4 preview using the desktop viewport.')
    project_arg(preview)
    preview.add_argument('--output', type=Path, required=True)
    preview.add_argument('--frames', type=int, help='Render up to this many frames from the saved trim.')
    camera_refinement_flag(preview, 'Show the saved camera with or without background refinement (FlowHMR / ExMR).')
    json_option(preview)

    stream = commands.add_parser('stream', help='Stream saved motion to Blender or Unreal.')
    project_arg(stream)
    take_flags(stream)
    stream.add_argument('--target', choices=('blender','unreal'), required=True)
    stream.add_argument('--port', type=int, help='Blender TCP port.')
    stream.add_argument('--wait', type=float, default=2., help='Seconds for the receiver to connect.')
    stream.add_argument('--seconds', type=float, help='Optional time limit, including for loops.')
    stream.add_argument('--loop', action='store_true')
    stream.add_argument('--face', action='store_true', help='Send tracked face curves to Unreal.')
    json_option(stream)
