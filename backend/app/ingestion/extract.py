from dataclasses import dataclass,field
import re
import subprocess


@dataclass
class Segment:
    text: str
    metadata: dict=field(default_factory=dict)


def clean(text):
    text=text.replace('\x00','').replace('\r\n','\n').replace('\r','\n')
    text=re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]',' ',text)
    return re.sub(r'\n{3,}','\n\n',text).strip()


def html_segments(raw):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(raw,'html.parser')
    for tag in soup.select('script,style,nav,header,footer,noscript'):tag.decompose()
    messages=soup.select('.message.default')
    if messages:
        result=[];last_author=''
        for node in messages:
            content=node.select_one('.text')
            if content is None:continue
            text=clean(content.get_text('\n',strip=True))
            if not text:continue
            author=node.select_one('.from_name')
            if author:last_author=author.get_text(' ',strip=True)
            date=node.select_one('.date')
            stamp=(date.get('title') or date.get_text(' ',strip=True)) if date else ''
            prefix=' · '.join(value for value in [stamp,last_author] if value)
            result.append(Segment((prefix+'\n' if prefix else '')+text,{
                'message_id':node.get('id'),'author':last_author,'date':stamp,
            }))
        return result
    archive_messages=soup.select('.message')
    if archive_messages:
        result=[]
        for i,node in enumerate(archive_messages):
            header=node.select_one('.message__header')
            heading=header.get_text(' ',strip=True) if header else ''
            if header:header.decompose()
            for service in node.select('.kludges'):service.decompose()
            text=clean(node.get_text('\n',strip=True))
            if not text:continue
            parent=node.find_parent(class_='item')
            message_id=node.get('id') or (parent.get('id') if parent else None) or f'message-{i}'
            result.append(Segment((heading+'\n' if heading else '')+text,
                                  {'message_id':message_id,'header':heading}))
        return result
    main=soup.find('main') or soup.find('article') or soup.body or soup
    return [Segment(clean(main.get_text('\n',strip=True)))]


def extract(path,cfg):
    suffix=path.suffix.lower()
    if suffix=='.pdf':
        from pypdf import PdfReader
        reader=PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(''):raise ValueError('Encrypted PDF needs a password')
        result=[Segment(clean(page.extract_text() or ''),{'page':i+1}) for i,page in enumerate(reader.pages)]
    elif suffix=='.docx':
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        result=[];section=''
        for block in Document(str(path)).iter_inner_content():
            if isinstance(block,Paragraph):
                if block.style and block.style.name.startswith(('Heading','Заголовок')):section=block.text.strip()
                text=block.text
            elif isinstance(block,Table):
                rows=[]
                for row in block.rows:
                    cells=[];seen=set()
                    for cell in row.cells:
                        if cell._tc in seen:continue
                        seen.add(cell._tc);cells.append(clean(cell.text).replace('\n','; '))
                    rows.append(' | '.join(cells))
                text='\n'.join(rows)
            else:continue
            result.append(Segment(clean(text),{'section':section}))
    elif suffix=='.doc':
        result=subprocess.run([
            'docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--mount',f'type=bind,source={path.resolve()},target=/input.doc,readonly',
            cfg.get('doc_converter_image','rag-doc-converter:local'),'/input.doc',
        ],capture_output=True,timeout=120)
        if result.returncode:raise ValueError('DOC converter failed; build tools/doc-converter image. '+result.stderr.decode('utf-8','replace')[:250])
        result=[Segment(clean(result.stdout.decode('utf-8')))]
    elif suffix in ('.html','.htm'):result=html_segments(path.read_bytes())
    else:
        raw=path.read_bytes()
        text=raw.decode('utf-8-sig') if not raw.startswith((b'\xff\xfe',b'\xfe\xff')) else raw.decode('utf-16')
        result=[Segment(clean(text))]
    result=[s for s in result if s.text.strip()]
    if not result:raise ValueError('No extractable text; image-only PDF requires OCR')
    return result
