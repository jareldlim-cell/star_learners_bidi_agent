/**
 * kb.js - handles the standalone Knowledge Base management page.
 */

async function postJson(path, body) {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return resp.json();
}

// upload
const kbUploadForm = document.getElementById('kbUploadForm');
const kbUploadUrl = document.getElementById('kbUploadUrl');
const kbUploadStatus = document.getElementById('kbUploadStatus');

kbUploadForm.addEventListener('submit', async evt => {
  evt.preventDefault();
  const url = kbUploadUrl.value.trim();
  if (!url) return;
  kbUploadStatus.textContent = 'Uploading...';
  try {
    const data = await postJson('/upload', { url });
    if (data.status) {
      kbUploadStatus.textContent = 'Import Completed';
    } else {
      kbUploadStatus.textContent = 'Error: ' + (data.error || JSON.stringify(data));
    }
  } catch (e) {
    kbUploadStatus.textContent = 'Exception: ' + e;
  }
});

// list files
const kbListFilesBtn = document.getElementById('kbListFiles');
const kbFilesList = document.getElementById('kbFilesList');

kbListFilesBtn.addEventListener('click', async () => {
  kbFilesList.textContent = 'Loading...';
  try {
    const res = await fetch('/kb/files');
    const data = await res.json();
    if (data.files) {
      kbFilesList.innerHTML = '';
      data.files.forEach(f => {
        const div = document.createElement('div');
        div.textContent = `${f.id} (${f.source})`;
        kbFilesList.appendChild(div);
      });
    } else {
      kbFilesList.textContent = 'Error: ' + (data.error || JSON.stringify(data));
    }
  } catch (e) {
    kbFilesList.textContent = 'Exception: ' + e;
  }
});

// delete file
const kbDeleteFileBtn = document.getElementById('kbDeleteFile');
const kbDeleteFileId = document.getElementById('kbDeleteFileId');
const kbDeleteStatus = document.getElementById('kbDeleteStatus');

kbDeleteFileBtn.addEventListener('click', async () => {
  const id = kbDeleteFileId.value.trim();
  if (!id) return;
  kbDeleteStatus.textContent = 'Deleting...';
  try {
    const data = await postJson('/kb/delete_file', { file_id: id });
    if (data.status) kbDeleteStatus.textContent = 'Deleted';
    else kbDeleteStatus.textContent = 'Error: '+(data.error||JSON.stringify(data));
  } catch (e) {
    kbDeleteStatus.textContent = 'Exception: '+e;
  }
});

// delete corpus
const kbDeleteCorpusBtn = document.getElementById('kbDeleteCorpus');
const kbDeleteCorpusStatus = document.getElementById('kbDeleteCorpusStatus');

kbDeleteCorpusBtn.addEventListener('click', async () => {
  if (!confirm('Really delete entire corpus? This cannot be undone.')) return;
  kbDeleteCorpusStatus.textContent = 'Deleting...';
  try {
    const data = await postJson('/kb/delete_corpus', { confirm: 'DELETE' });
    if (data.status) kbDeleteCorpusStatus.textContent = 'Corpus deleted';
    else kbDeleteCorpusStatus.textContent = 'Error: '+(data.error||JSON.stringify(data));
  } catch (e) {
    kbDeleteCorpusStatus.textContent = 'Exception: '+e;
  }
});
