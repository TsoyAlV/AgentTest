import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from docx import Document
from pypdf import PdfWriter

from app.config import Settings
from app.embeddings import Embeddings
from app.errors import ServiceError
from app.ingestion.chunking import chunk_segments
from app.ingestion.extract import Segment,extract,html_segments
from app.ingestion.files import scan,file_hash,classify,FileEntry


class WordTokenizer:
    def encode(self,text,add_special_tokens=False):
        return SimpleNamespace(offsets=[m.span() for m in re.finditer(r'\S+',text)])


def test_scan_mapping_hashes_and_unknown_folders(tmp_path):
    folder=tmp_path/'employee';folder.mkdir()
    a=folder/'a.txt';a.write_text('same',encoding='utf-8')
    b=folder/'renamed.txt';b.write_text('same',encoding='utf-8')
    (tmp_path/'unmapped.txt').write_text('private',encoding='utf-8')
    entries,errors=scan({'root':tmp_path,'folders':{'employee':['employee']}})
    assert errors==[]
    good=[e for e in entries if not e.error]
    assert len(good)==2 and good[0].digest==good[1].digest
    assert good[0].roles==['employee']
    assert any(e.error=='Unmapped top-level folder' for e in entries)
    assert classify(good[1],None,True)=='duplicate'


def test_reconciliation_distinguishes_pipeline_acl_and_content(tmp_path):
    file=tmp_path/'a.txt';file.write_text('one')
    e=FileEntry(file,'employee/a.txt',['employee'],file_hash(file),3)
    old={'indexed_hash':e.digest,'pipeline_signature':'v1','role_names':['employee'],'status':'indexed'}
    assert classify(e,old,signature='v1')=='unchanged'
    assert classify(e,old,signature='v2')=='changed'
    assert classify(e,{**old,'role_names':['hr']},signature='v1')=='permissions_changed'
    assert classify(e,{**old,'status':'missing'},signature='v1')=='permissions_changed'
    assert classify(e,old,signature='v1',force=True)=='changed'


def test_docx_keeps_paragraph_table_order(tmp_path):
    doc=Document();doc.add_heading('Раздел',level=1);doc.add_paragraph('До таблицы')
    table=doc.add_table(rows=2,cols=2)
    for row,values in zip(table.rows,[['Возраст','Норматив'],['10 лет','20 раз']]):
        for cell,value in zip(row.cells,values):cell.text=value
    doc.add_paragraph('После таблицы')
    path=tmp_path/'sample.docx';doc.save(path)
    segments=extract(path,{})
    text='\n'.join(s.text for s in segments)
    assert text.index('До таблицы')<text.index('10 лет | 20 раз')<text.index('После таблицы')
    assert segments[-1].metadata['section']=='Раздел'


def test_empty_pdf_reports_ocr_requirement(tmp_path):
    path=tmp_path/'scan.pdf';pdf=PdfWriter();pdf.add_blank_page(100,100)
    with path.open('wb') as file:pdf.write(file)
    with pytest.raises(ValueError,match='OCR'):extract(path,{})


def test_archive_messages_exclude_navigation_and_keep_headers():
    html='''<nav>Next page</nav><div class="message"><div class="message__header">Автор, 20.11</div>
      Текст сообщения<div class="kludges">service</div></div><div class="message">
      <div class="message__header">Другой автор</div>Второе сообщение</div>'''
    segments=html_segments(html)
    assert len(segments)==2
    assert 'Автор, 20.11' in segments[0].text
    assert 'Next page' not in str(segments) and 'service' not in str(segments)
    assert segments[0].metadata['message_id']


def test_telegram_joined_messages_keep_author():
    html='''<div class="message default" id="message1"><div class="from_name">Автор</div><div class="text">Первое</div></div>
    <div class="message default joined" id="message2"><div class="text">Второе</div></div>'''
    segments=html_segments(html)
    assert len(segments)==2 and segments[1].metadata['author']=='Автор'


def test_token_windows_overlap_and_page_metadata():
    segments=[Segment('один два три четыре',{'page':1}),Segment('пять шесть семь восемь',{'page':2})]
    chunks=chunk_segments(segments,WordTokenizer(),size=5,overlap=2)
    assert all(c.metadata['token_count']<=5 for c in chunks)
    assert chunks[0].content.split()[-2:]==chunks[1].content.split()[:2]
    assert chunks[0].metadata['page_start']==1 and chunks[0].metadata['page_end']==2
    assert chunks[-1].content.endswith('восемь')


def test_embedding_batches_follow_response_indexes():
    def handler(request):
        assert json.loads(request.content)['input']==['one','two']
        return httpx.Response(200,json={'data':[{'index':1,'embedding':[2.]*1024},{'index':0,'embedding':[1.]*1024}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        vectors=Embeddings(Settings(_env_file=None,embedding_base_url='http://embedding/v1'),http).encode_many(['one','two'])
    assert vectors.shape==(2,1024) and vectors[0,0]==1 and vectors[1,0]==2


def test_wrong_batch_count_is_rejected():
    with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'data':[]}))) as http:
        with pytest.raises(ServiceError):Embeddings(Settings(_env_file=None,embedding_base_url='http://embedding/v1'),http).encode_many(['one'])
