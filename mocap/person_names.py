"""Display names and portable export filenames; stable tracking IDs never change."""
import re
import unicodedata
from pathlib import Path


def validate_name(value):
    if not isinstance(value,str):raise ValueError('Enter a person name.')
    value=value.strip()
    if not value:raise ValueError('The name cannot be empty.')
    if any(unicodedata.category(c)=='Cc' for c in value):
        raise ValueError('The name cannot contain control characters.')
    if len(value.encode('utf-8'))>60:
        raise ValueError('The name is too long (maximum 60 UTF-8 bytes).')
    return value


def export_stem(name):
    name=validate_name(name)
    stem=re.sub(r'[<>:"/\\|?*]', '_', name).strip(' .') or 'Person'
    if stem.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'{p}{n}' for p in ('COM','LPT') for n in range(1,10))}:
        stem='_'+stem
    return stem


def export_path(folder,stem,used,suffix='.fbx'):
    """Reserve names across both this batch and existing files on Windows."""
    candidate=Path(folder)/(stem+suffix);number=2
    while candidate.name.casefold() in used or candidate.exists():
        candidate=Path(folder)/(stem+f'_{number}'+suffix);number+=1
    used.add(candidate.name.casefold())
    return candidate
