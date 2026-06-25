import { useRef, useState } from 'react'

const FORMATS  = ['MP4', 'AVI', 'MOV', 'MKV', 'WEBM']
const MAX_MB   = 500
const ACCEPT   = '.mp4,.avi,.mov,.mkv,.webm'

export default function Upload({ onUpload }) {
  const inputRef         = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [info,     setInfo]     = useState(null)   // last uploaded video info

  async function handleFile(file) {
    setError(null)
    if (!file) return

    const ext = file.name.split('.').pop().toLowerCase()
    if (!FORMATS.map(f => f.toLowerCase()).includes(ext)) {
      setError(`Unsupported format .${ext}. Accepted: ${FORMATS.join(', ')}`)
      return
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(`File is ${(file.size / 1024 / 1024).toFixed(1)} MB — limit is ${MAX_MB} MB`)
      return
    }

    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res  = await fetch('/api/upload', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setInfo(data)
      onUpload({ info: data, file })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div>
      <div className="limits">
        <span><strong>Formats:</strong> {FORMATS.join(' · ')}</span>
        <span><strong>Max size:</strong> {MAX_MB} MB</span>
        <span><strong>Memory:</strong> full video decoded into RAM — ~1–4 GB for a 100 MB clip</span>
      </div>

      <div
        className={`dropzone${dragging ? ' drag-over' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          onChange={e => handleFile(e.target.files[0])}
        />
        <div className="dropzone-icon">🎬</div>
        {loading
          ? <div>Uploading and decoding frames…</div>
          : <div>Drop video here or <a>browse</a></div>
        }
        <div className="dropzone-hint">{FORMATS.join(', ')} · up to {MAX_MB} MB</div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {info && !loading && (
        <div className="meta-grid">
          <span><strong>{info.totalFrames}</strong> frames</span>
          <span><strong>{info.width}×{info.height}</strong></span>
          <span><strong>{info.fps}</strong> fps</span>
          <span><strong>{info.duration.toFixed(1)}s</strong></span>
        </div>
      )}
    </div>
  )
}
