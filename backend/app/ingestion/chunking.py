from dataclasses import dataclass
import hashlib
import json
from importlib.metadata import version
from pathlib import Path


@dataclass
class Chunk:
    content: str
    section_title: str
    metadata: dict


def load_tokenizer(cfg):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    location=hf_hub_download(cfg['tokenizer_model'],'tokenizer.json',revision=cfg.get('tokenizer_revision','main'),
                             cache_dir=cfg['tokenizer_cache'])
    tokenizer=Tokenizer.from_file(location)
    tokenizer.no_truncation();tokenizer.no_padding()
    # Fingerprint the actual vocabulary/config, even if the configured revision is main.
    tokenizer_digest=hashlib.sha256(Path(location).read_bytes()).hexdigest()
    return tokenizer,tokenizer_digest


def signature(cfg,settings,tokenizer_digest):
    data={'parser':'files-v1','chunk_tokens':cfg['chunk_tokens'],'overlap_tokens':cfg['overlap_tokens'],
          'tokenizer':tokenizer_digest,'embedding_model':settings.embedding_model,
          'dimensions':settings.embedding_dimensions,'profile':cfg.get('embedding_profile',''),
          'libraries':{name:version(name) for name in ['pypdf','python-docx','beautifulsoup4','tokenizers']}}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def chunk_segments(segments,tokenizer,size=500,overlap=80):
    text='';ranges=[]
    for segment in segments:
        if text:text+='\n\n'
        start=len(text);text+=segment.text;ranges.append((start,len(text),segment.metadata))
    encoding=tokenizer.encode(text,add_special_tokens=False)
    offsets=[(a,b) for a,b in encoding.offsets if b>a]
    if not offsets:raise ValueError('Tokenizer returned no text tokens')
    result=[]
    for index in range(0,len(offsets),size-overlap):
        stop=min(index+size,len(offsets));start_char=offsets[index][0];end_char=offsets[stop-1][1]
        content=text[start_char:end_char].strip()
        related=[meta for a,b,meta in ranges if a<end_char and b>start_char]
        meta={'token_count':stop-index,'char_start':start_char,'char_end':end_char}
        pages=[m['page'] for m in related if m.get('page')]
        if pages:meta.update(page_start=min(pages),page_end=max(pages))
        messages=[m['message_id'] for m in related if m.get('message_id')]
        if messages:meta['message_ids']=messages[:200]
        section=next((m['section'] for m in related if m.get('section')),'')
        if not section and pages:section=f'Страницы {min(pages)}–{max(pages)}'
        if content:result.append(Chunk(content,section,meta))
        if stop==len(offsets):break
    return result
