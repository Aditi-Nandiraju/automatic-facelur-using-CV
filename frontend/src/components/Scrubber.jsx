import { useEffect, useRef, useState, useCallback } from 'react'

export default function Scrubber({ videoInfo, videoFile, onSelect }) {
  const { videoId, fps, totalFrames } = videoInfo

  const videoRef     = useRef(null)
  const canvasRef    = useRef(null)
  const snapshotRef  = useRef(null)
  const videoUrl     = useRef(null)
  const busyRef      = useRef(false)   // prevent concurrent detections

  const [phase,     setPhase]     = useState('player')  // 'player' | 'select'
  const [boxes,     setBoxes]     = useState([])
  const [frameIdx,  setFrameIdx]  = useState(null)
  const [selBox,    setSelBox]    = useState(null)
  const [detecting, setDetecting] = useState(false)
  const [error,     setError]     = useState(null)

  // Create blob URL once for the file
  useEffect(() => {
    const url = URL.createObjectURL(videoFile)
    videoUrl.current = url
    setPhase('player')
    setBoxes([])
    setSelBox(null)
    setError(null)
    snapshotRef.current = null
    return () => URL.revokeObjectURL(url)
  }, [videoFile])

  // Redraw canvas whenever boxes or selection changes
  useEffect(() => {
    if (snapshotRef.current) drawOverlay(boxes, selBox)
  }, [boxes, selBox])

  // Auto-detect whenever the video is paused
  const handleVideoPause = useCallback(async () => {
    if (busyRef.current) return
    const video = videoRef.current
    if (!video || video.ended) return

    busyRef.current = true
    const idx = Math.min(totalFrames - 1, Math.max(0, Math.round(video.currentTime * fps)))
    setFrameIdx(idx)
    setPhase('select')
    setDetecting(true)
    setBoxes([])
    setSelBox(null)
    setError(null)
    snapshotRef.current = null

    try {
      // Fetch frame image and face boxes in parallel
      const [frameRes, facesRes] = await Promise.all([
        fetch(`/api/frame/${videoId}/${idx}`),
        fetch(`/api/faces/${videoId}/${idx}`),
      ])

      // Always render the frame — even if detection fails the image shows
      if (!frameRes.ok) throw new Error(`Could not load frame ${idx} (${frameRes.status})`)
      const blob = await frameRes.blob()
      const img  = new Image()
      img.src    = URL.createObjectURL(blob)
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej })

      const canvas  = canvasRef.current
      canvas.width  = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)
      snapshotRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height)
      URL.revokeObjectURL(img.src)

      // Face detection
      if (!facesRes.ok) {
        const t = await facesRes.text()
        let msg
        try { msg = JSON.parse(t).detail } catch { msg = t }
        setError(msg || `Server error ${facesRes.status}`)
      } else {
        const data = await facesRes.json()
        setBoxes(data.boxes || [])
      }
    } catch (e) {
      setError(e.message)
    }

    setDetecting(false)
    busyRef.current = false
  }, [videoId, fps, totalFrames])

  function resumeVideo() {
    setPhase('player')
    setBoxes([])
    setSelBox(null)
    setError(null)
    videoRef.current?.play()
  }

  function drawOverlay(boxList, sel) {
    const canvas = canvasRef.current
    if (!canvas || !snapshotRef.current) return
    const ctx = canvas.getContext('2d')
    ctx.putImageData(snapshotRef.current, 0, 0)
    if (!boxList.length) return

    const lw = Math.max(2, canvas.width / 300)
    const fs = Math.max(13, canvas.width / 55)

    boxList.forEach((b, i) => {
      const [x1, y1, x2, y2] = b
      const chosen = sel && b.every((v, j) => v === sel[j])
      const color  = chosen ? '#facc15' : '#22c55e'

      // Subtle fill on selected box
      if (chosen) {
        ctx.fillStyle = 'rgba(250,204,21,0.18)'
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1)
      }

      // Border
      ctx.strokeStyle = color
      ctx.lineWidth   = chosen ? lw * 3 : lw * 1.5
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      // Label pill above box
      const label = chosen ? '✓ Selected' : `Face ${i + 1}`
      ctx.font     = `bold ${fs}px sans-serif`
      const tw     = ctx.measureText(label).width
      const pad    = fs * 0.3
      const lx     = x1
      const ly     = Math.max(y1 - fs - pad * 2, 0)

      ctx.fillStyle = color
      ctx.fillRect(lx, ly, tw + pad * 2, fs + pad * 2)
      ctx.fillStyle = '#000'
      ctx.fillText(label, lx + pad, ly + fs + pad * 0.5)
    })
  }

  function handleCanvasClick(e) {
    if (!boxes.length || detecting) return
    const canvas = canvasRef.current
    if (!canvas || canvas.width === 0) return

    const rect = canvas.getBoundingClientRect()
    const sx   = canvas.width  / rect.width
    const sy   = canvas.height / rect.height
    const x    = (e.clientX - rect.left) * sx
    const y    = (e.clientY - rect.top)  * sy

    const hits = boxes.filter(b => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
    const picked = hits.length
      ? hits.reduce((a, b) => area(a) < area(b) ? a : b)
      : boxes.reduce((a, b) => dist(a, x, y) < dist(b, x, y) ? a : b)

    setSelBox(picked)
    onSelect({ frameIdx, box: picked })
  }

  return (
    <div>
      {/* Video player */}
      <video
        ref={videoRef}
        src={videoUrl.current || ''}
        controls
        onPause={handleVideoPause}
        style={{
          width: '100%', borderRadius: 6, background: '#000',
          display: phase === 'player' ? 'block' : 'none',
        }}
      />

      {/* Canvas — frozen frame + face boxes */}
      <div style={{ display: phase === 'select' ? 'block' : 'none', position: 'relative' }}>
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          style={{
            display: 'block', width: '100%', borderRadius: 6,
            cursor: boxes.length && !detecting ? 'crosshair' : 'default',
          }}
        />
        {detecting && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.5)', borderRadius: 6,
          }}>
            <span style={{ color: '#fff', fontWeight: 600, fontSize: '1.1rem' }}>
              Detecting faces…
            </span>
          </div>
        )}
      </div>

      {/* Status row */}
      {phase === 'player' && (
        <p className="canvas-hint" style={{ marginTop: '.5rem' }}>
          Pause on a frame with a clear view of the face you want to blur.
          Faces will be detected automatically.
        </p>
      )}

      {phase === 'select' && !detecting && (
        <div style={{ marginTop: '.6rem', display: 'flex', alignItems: 'center', gap: '.75rem', flexWrap: 'wrap' }}>
          {error ? (
            <span className="error-box">{error}</span>
          ) : boxes.length === 0 ? (
            <span className="canvas-hint">No faces detected — try a different frame.</span>
          ) : !selBox ? (
            <span className="canvas-hint">
              {boxes.length} face{boxes.length > 1 ? 's' : ''} detected —
              click a <strong style={{ color: '#22c55e' }}>green box</strong> to select one.
            </span>
          ) : (
            <span style={{ color: '#22c55e', fontWeight: 600 }}>
              ✓ Face selected at frame {frameIdx}
            </span>
          )}
          <button
            className="btn btn-primary"
            style={{ marginLeft: 'auto' }}
            onClick={resumeVideo}
          >
            ← Try a different frame
          </button>
        </div>
      )}
    </div>
  )
}

const area = b => (b[2] - b[0]) * (b[3] - b[1])
const dist = (b, x, y) => ((b[0] + b[2]) / 2 - x) ** 2 + ((b[1] + b[3]) / 2 - y) ** 2
